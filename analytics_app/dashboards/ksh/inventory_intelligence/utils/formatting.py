"""
Display formatting helpers and design-system constants.
"""

from __future__ import annotations

from typing import Optional


# ── Brand palette ─────────────────────────────────────────────────────────────

COLOR_PRIMARY    = "#0F6E56"   # teal  — healthy / positive
COLOR_AMBER      = "#854F0B"   # amber — warning
COLOR_RED        = "#A32D2D"   # red   — critical / stockout
COLOR_DEEP_RED   = "#791F1F"   # deep red — negative stock
COLOR_NEUTRAL    = "#888780"   # grey  — unknown / inactive
COLOR_BLUE       = "#0C447C"   # blue  — standard / informational

COLOR_BG         = "#F5F6FA"
COLOR_SURFACE    = "#FFFFFF"
COLOR_BORDER     = "#E5E7EB"
COLOR_TEXT       = "#1A1A2E"
COLOR_TEXT_MUTED = "#6B7280"


# ── Stock status ─────────────────────────────────────────────────────────────

STATUS_COLORS: dict[str, str] = {
    "adequate":  COLOR_PRIMARY,    # teal   — healthy
    "low":       "#D97706",        # amber  — watch
    "critical":  "#DC2626",        # red    — order now
    "zero":      "#991B1B",        # dark red — stocked out
    "stockout":  "#991B1B",        # dark red — KPI-path alias for zero
    "negative":  COLOR_DEEP_RED,   # deepest — negative SOH
}

STATUS_EMOJI: dict[str, str] = {
    "adequate":  "🟢",
    "low":       "🟡",
    "critical":  "🔴",
    "zero":      "🔴",
    "negative":  "⛔",
}

STATUS_ORDER = ["negative", "zero", "critical", "low", "adequate"]

DOS_COLORS: dict[str, str] = {
    "red":   COLOR_RED,
    "amber": COLOR_AMBER,
    "green": COLOR_PRIMARY,
}


# ── Action colors ─────────────────────────────────────────────────────────────

ACTION_COLORS: dict[str, str] = {
    "ORDER NOW":       COLOR_RED,
    "ORDER THIS WEEK": COLOR_AMBER,
    "MONITOR":         COLOR_BLUE,
    "REVIEW":          COLOR_NEUTRAL,
}

ACTION_ICONS: dict[str, str] = {
    "ORDER NOW":       "🔴",
    "ORDER THIS WEEK": "🟡",
    "MONITOR":         "🔵",
    "REVIEW":          "⚪",
}


# ── Priority ──────────────────────────────────────────────────────────────────

PRIORITY_COLORS: dict[str, str] = {
    "CRITICAL": COLOR_RED,
    "HIGH":     COLOR_AMBER,
    "STANDARD": COLOR_BLUE,
}

# ── Confidence ───────────────────────────────────────────────────────────────

CONFIDENCE_COLORS: dict[str, str] = {
    "HIGH":   COLOR_PRIMARY,
    "MEDIUM": COLOR_AMBER,
    "LOW":    COLOR_NEUTRAL,
}


# ── Number formatters ─────────────────────────────────────────────────────────

def fmt_kes(value: Optional[float], decimals: int = 0) -> str:
    if value is None:
        return "—"
    return f"KES {value:,.{decimals}f}"


def fmt_kes_millions(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"KES {value / 1_000_000:.1f}M"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}%"


def fmt_days(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{int(value)}d"


def fmt_delta(value: Optional[float], suffix: str = "%") -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{suffix}"


# ── Drug name cleaner ─────────────────────────────────────────────────────────

def fmt_drug_name(name) -> str:
    """
    Clean a canonical drug name for display.
    - None / empty / "None" string → "—"
    - Pure numeric string (product ID stored as name in taxonomy) → "[#34449]"
    - Otherwise return as-is.
    """
    if name is None:
        return "—"
    s = str(name).strip()
    if s.lower() in ("", "none", "null"):
        return "—"
    if s.isdigit():
        return f"[#{s}]"
    return s


def clean_drug_names(df, col: str = "CANONICAL_NAME"):
    """Apply fmt_drug_name to a CANONICAL_NAME column in-place. Safe if col is missing."""
    import pandas as pd
    if col in df.columns:
        df = df.copy()
        df[col] = df[col].map(fmt_drug_name)
    return df


# ── Lookup helpers ────────────────────────────────────────────────────────────

def status_color(status: Optional[str]) -> str:
    return STATUS_COLORS.get(str(status or "").lower(), COLOR_NEUTRAL)


def status_emoji(status: Optional[str]) -> str:
    return STATUS_EMOJI.get(str(status or "").lower(), "⚪")


def priority_color(priority: Optional[str]) -> str:
    return PRIORITY_COLORS.get(str(priority or "").upper(), COLOR_NEUTRAL)


def action_color(action: Optional[str]) -> str:
    return ACTION_COLORS.get(str(action or "").upper(), COLOR_NEUTRAL)


def confidence_color(confidence: Optional[str]) -> str:
    return CONFIDENCE_COLORS.get(str(confidence or "").upper(), COLOR_NEUTRAL)
