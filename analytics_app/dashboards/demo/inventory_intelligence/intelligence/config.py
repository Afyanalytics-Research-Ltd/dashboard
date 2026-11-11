"""
Centralised clinical and operational configuration.
All thresholds and classification rules live here — nothing scattered across business logic.
"""

# ── Clinical Priority ─────────────────────────────────────────────────────────

CRITICAL_THERAPEUTIC_SUBCLASSES: tuple = (
    "Opioid Analgesics",
    "Carbapenems",
    "Glycopeptides",
    "Oxazolidinones",
    "Uterotonics",
    "Antiretrovirals",
)

HIGH_THERAPEUTIC_CLASSES: tuple = (
    "Cardiovascular",
    "Endocrine & Metabolic",
    "Antimicrobials",
    "Neurological",
    "Antiepileptics",
    "Oncology",
)

# ── Stock Level Thresholds (days of stock) ────────────────────────────────────

DOS_CRITICAL: int = 7
DOS_LOW: int = 30

# ── Model Data Requirements ───────────────────────────────────────────────────

MIN_DAYS_FOR_FORECAST: int = 14
MIN_MONTHS_HIGH_CONFIDENCE: int = 12    # legacy month-based (kept for compatibility)
MIN_MONTHS_MEDIUM_CONFIDENCE: int = 3   # legacy month-based (kept for compatibility)

# Confidence scoring thresholds (days-based)
MIN_DAYS_HIGH_CONFIDENCE: int   = 90    # ≥90 days required for HIGH
MIN_DAYS_MEDIUM_CONFIDENCE: int = 30    # ≥30 days required for MEDIUM

# Syntetos-Boylan demand pattern classification
# ADI    = total calendar days / non-zero dispensing days  (computed on DAILY data)
# CV²_nz = squared CV of non-zero quantities (computed on WEEKLY aggregated data — see note below)
#
#              CV²_nz < 0.49      CV²_nz ≥ 0.49
#  ADI < 7.0   SMOOTH             ERRATIC
#  ADI ≥ 7.0   INTERMITTENT       LUMPY
#
# ADI threshold rationale: Original SB value of 1.32 was derived for weekly/monthly industrial
# data. At daily pharmacy granularity, ADI=1.32 means "dispensed 76% of calendar days" — almost
# every drug qualifies as INTERMITTENT or LUMPY. ADI=7.0 maps to once-per-week dispensing, the
# pharmacologically meaningful boundary between regularly-stocked and episodic-demand drugs in a
# private hospital context. Validated against exposure analysis: with 1.32, 76% of the catalog
# fell LUMPY/LOW; with 7.0 this should resolve to a realistic split.
#
# CV²_nz weekly aggregation rationale: Private hospital dispensing in 30-day or 90-day supplies
# creates large daily quantity variance that reflects prescription batch size, not genuine demand
# variability. Aggregating to weekly totals before computing CV_nz absorbs this noise without
# losing signal on true quantity volatility. ADI is kept on daily data so the threshold (7.0)
# remains interpretable in calendar-day units.
#
# Confidence mapping:
#   SMOOTH       + ≥90d → HIGH
#   SMOOTH       + 30-89d → MEDIUM
#   ERRATIC      + ≥30d → MEDIUM  (frequent but quantity varies — trend is still useful)
#   INTERMITTENT + ≥30d → MEDIUM  (infrequent but consistent quantity per event)
#   LUMPY                → LOW    (infrequent AND variable — genuinely unpredictable)
#   Any type     + <30d  → LOW    (insufficient history)
SB_ADI_THRESHOLD: float = 7.0   # calibrated for daily pharmacy data; weekly dispensing = regular demand boundary
SB_CV2_THRESHOLD: float = 0.49

# ── Lead Time ─────────────────────────────────────────────────────────────────

DEFAULT_LEAD_TIME_DAYS: int = 14
MIN_LEAD_TIME_DAYS: int = 1
MAX_LEAD_TIME_DAYS: int = 180
MIN_LEAD_TIME_OBSERVATIONS: int = 3

# ── Safety Stock / Reorder ───────────────────────────────────────────────────

SERVICE_LEVEL_Z: dict = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}
DEFAULT_SERVICE_LEVEL: float = 0.95
DEFAULT_ORDER_COST_KES: float = 500.0
DEFAULT_HOLDING_RATE: float = 0.25   # 25% of unit cost per year

# ── Demand Engine ────────────────────────────────────────────────────────────

EWM_SPAN_MIN: int = 7
EWM_SPAN_MAX: int = 30
TREND_THRESHOLD: float = 0.15        # >15% change = directional trend

# ── Anomaly Detection ─────────────────────────────────────────────────────────

ANOMALY_Z_THRESHOLD: float = 2.0
ANOMALY_BASELINE_DAYS: int = 90
ANOMALY_RECENT_DAYS: int = 14

# ── ABC Analysis ─────────────────────────────────────────────────────────────

ABC_A_CUM_PCT: float = 0.70
ABC_B_CUM_PCT: float = 0.90

# ── Dead / Slow Stock ────────────────────────────────────────────────────────

SLOW_MOVING_DAYS: int = 30
DEAD_STOCK_DAYS: int = 90

# ── Redistribution ───────────────────────────────────────────────────────────

SURPLUS_DOS_THRESHOLD: int = 90
SURPLUS_IDLE_THRESHOLD_DAYS: int = 30
REDISTRIBUTION_TARGET_DAYS: int = 30
REDISTRIBUTION_BUFFER_DAYS: int = 30
