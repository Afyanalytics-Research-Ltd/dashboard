"""Thresholds, palette and reference data. One place for every tunable rule.

Clinical and operational thresholds for the inventory dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Stock-level bands (days of stock) ──────────────────────────────────
DOS_CRITICAL = 7
DOS_LOW = 30

# ── Demand model ────────────────────────────────────────────────────────
MIN_DAYS_FOR_FORECAST = 14
MIN_DAYS_HIGH_CONFIDENCE = 90
MIN_DAYS_MEDIUM_CONFIDENCE = 30
SB_ADI_THRESHOLD = 7.0      # calibrated for daily pharmacy data
SB_CV2_THRESHOLD = 0.49
EWM_SPAN_MIN, EWM_SPAN_MAX = 7, 30
TREND_THRESHOLD = 0.15
TREND_MULT = {"UP": 1.10, "DOWN": 0.95, "STABLE": 1.00}
FORECAST_Z = 1.645          # 90% two-sided band

# ── Anomaly ─────────────────────────────────────────────────────────────
ANOMALY_Z = 2.0
ANOMALY_BASELINE_DAYS = 90
ANOMALY_RECENT_DAYS = 14

# ── Replenishment (lead time measured from real receipts) ─────────
DEFAULT_LEAD_TIME_DAYS = 14
DEFAULT_LEAD_TIME_STD = 4.2
MIN_LT_OBSERVATIONS = 3
# measured waits include slow reordering (items with > 30-day waits are restocked only
# every ~101 days), so an item's lead time is capped at this multiple of the facility's typical wait, and its
# lead-time spread at the facility's typical spread. Both bounds come from the data; replace with supplier lead
# times once Dynamics purchase-order dates are available.
LEAD_TIME_CAP_X_MEDIAN = 2.0

# Availability (service level): share of in-use store items with stock on hand, per day. The stock ledger has frequent
# breaks (opening <> previous closing) until April 2026, so the monthly trend starts in May 2026.
AVAILABILITY_DAYS = 90
RELIABLE_LEDGER_FROM = "2026-05-01"
# Excess stock: items still in use but holding more than this many days of cover (distinct from idle/dead stock).
EXCESS_COVER_DAYS = 180
TARGET_COVER_DAYS = 30
SERVICE_LEVEL_Z = {"90%": 1.282, "95%": 1.645, "99%": 2.326}

# ── Dead / slow stock ───────────────────────────────────────────────────
DEAD_DAYS = 90
SLOW_DAYS = 30

# ── Redistribution & repeat stockouts ─────────────────────────────────────────────────────────────
REDISTRIBUTE_SURPLUS_DOS = 90      # a store holding > 90 days of cover (or idle stock) is a donor
REDISTRIBUTE_KEEP_DAYS = 60        # donor keeps 60 days of its own demand
REPEAT_STOCKOUT_EPISODES = 3       # >= 3 episodes in 12 months = repeat offender
NEW_STOCKOUT_DAYS = 14

# ── Palette ─────────────────────────────────────────────
PRIMARY = "#0F6E56"
PRIMARY_LIGHT = "#1D9E75"
AMBER = "#D97706"
RED = "#DC2626"
DARK_RED = "#991B1B"
PURPLE = "#6D28D9"
GREY = "#9CA3AF"
INK = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"

STATUS_ORDER = ["Negative", "Stocked out", "Critical", "Low", "Adequate", "No recent demand"]
STATUS_COLORS = {
    "Negative": PURPLE, "Stocked out": DARK_RED, "Critical": RED,
    "Low": AMBER, "Adequate": PRIMARY_LIGHT, "No recent demand": GREY,
}
TIER_COLORS = {"CRITICAL": DARK_RED, "HIGH": AMBER, "STANDARD": GREY}
STREAM_COLORS = {"Dispensing": PRIMARY, "Ward use": "#60A5FA"}
CLASS_ORDER = ["Pharmaceutical", "Medical supply", "Laboratory supply", "Radiology supply",
               "Non-medical", "Unclassified"]


# ── Kisumu disease calendar (MoH Kenya IDSR) ──
@dataclass(frozen=True)
class SeasonalDisease:
    name: str
    peak_months: tuple
    keywords: tuple          # matched (case-insensitive) against subclass, class and product name
    multiplier: float
    note: str


DISEASE_CALENDAR = (
    SeasonalDisease("Malaria", (3, 4, 5, 10, 11),
                    ("antimalarial", "malaria", "artemether", "quinine", "lumefantrine", "antiparasitic"),
                    1.45, "Long rains (Mar–May) and short rains (Oct–Nov) around Lake Victoria."),
    SeasonalDisease("Diarrhoeal disease & cholera", (5, 6, 11, 12),
                    ("antidiarrh", "oral rehydration", "ors ", "metronidazole", "zinc sulphate", "zinc sulfate"),
                    1.35, "Waterborne disease 4–6 weeks after heavy rains."),
    SeasonalDisease("Respiratory infections", (7, 8, 9),
                    ("bronchodilator", "salbutamol", "ipratropium", "budesonide", "beclomethasone",
                     "prednisolone", "macrolide", "azithromycin", "leukotriene", "respiratory"),
                    1.25, "Cold, dusty dry season (Jul–Sep)."),
    SeasonalDisease("Typhoid", (6, 7),
                    ("fluoroquinolone", "ciprofloxacin", "ceftriaxone", "cefixime"),
                    1.30, "4–6 weeks after the long rains."),
    SeasonalDisease("Malaria-associated anaemia", (4, 5, 6, 11, 12),
                    ("blood transfusion", "haematology"),
                    1.30, "Lags the malaria peaks by 2–4 weeks."),
    SeasonalDisease("Skin & wound infections", (4, 5, 10, 11),
                    ("flucloxacillin", "cloxacillin", "dicloxacillin", "wound care", "antiseptic",
                     "povidone", "chlorhexidine", "topical antibiotic"),
                    1.20, "Rainy-season flooding."),
)
