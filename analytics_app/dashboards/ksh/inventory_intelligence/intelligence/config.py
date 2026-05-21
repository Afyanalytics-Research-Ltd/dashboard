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
MIN_MONTHS_HIGH_CONFIDENCE: int = 12
MIN_MONTHS_MEDIUM_CONFIDENCE: int = 3

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
