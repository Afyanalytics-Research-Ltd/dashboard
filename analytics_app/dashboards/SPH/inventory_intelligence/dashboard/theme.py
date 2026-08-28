"""Design system: palette, Plotly template, global CSS, and formatters.

A teal-forward clinical look (light and dark aware). Colour is assigned by job
— brand/positive (teal), warning (amber), critical (red), informational
(blue), neutral (grey) — never by fixed numeric cutoffs.
"""
from __future__ import annotations

import math
from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Brand palette ─────────────────────────────────────────────────────────────

TEAL = "#0F6E56"
TEAL_DARK = "#2FB48D"
AMBER = "#B45309"
AMBER_DARK = "#E8983B"
RED = "#B42318"
RED_DARK = "#F07167"
BLUE = "#0C6291"
BLUE_DARK = "#4BA3D3"
VIOLET = "#6D4AA7"
VIOLET_DARK = "#A78BE0"

# Single-series / "primary" data colour — a vivid teal that keeps the brand
# identity while clearing ≥3:1 contrast on each surface (validated).
SERIES_PRIMARY = "#0C8F6B"
SERIES_PRIMARY_DARK = "#2FB48D"

STATUS = {"good": "#0C8F6B", "warning": AMBER, "serious": "#C2410C", "critical": RED}
STATUS_DARK = {"good": TEAL_DARK, "warning": AMBER_DARK, "serious": "#F08A5D", "critical": RED_DARK}

# Categorical series colours — a vivid, teal-led order validated for colour-vision
# safety in BOTH modes (worst adjacent CVD ΔE 9.2 light / 9.4 dark ≥ 8 target;
# normal-vision ΔE 19.6 / 19.3 ≥ 15 floor). Assigned in fixed order, never
# cycled. Do not reorder without re-running scripts/validate_palette.js.
CATEGORICAL_LIGHT = ["#1BAF7A", "#EB6834", "#2A78D6", "#EDA100",
                     "#E87BA4", "#008300", "#4A3AA7", "#E34948"]
CATEGORICAL_DARK = ["#199E70", "#D95926", "#3987E5", "#C98500",
                    "#D55181", "#008300", "#9085E9", "#E66767"]

SEQUENTIAL_TEAL = [
    "#d6efe6", "#b3e0d1", "#8fd0bb", "#69c0a5", "#41b090", "#1f9e7b",
    "#0f8968", "#0f6e56", "#0d5c48", "#0a4a3a", "#08392d",
]

_CHROME = {
    "light": {
        "surface": "#ffffff", "surface_2": "#f7f8fa", "page": "#f5f6fa",
        "ink": "#14181f", "ink_secondary": "#4b5563", "muted": "#8a919e",
        "border": "#e6e8ec", "grid": "#eceef1", "baseline": "#d3d6db",
    },
    "dark": {
        "surface": "#1a1d24", "surface_2": "#20242d", "page": "#14171d",
        "ink": "#f2f4f7", "ink_secondary": "#c3c8d2", "muted": "#8a919e",
        "border": "#2c313a", "grid": "#262b33", "baseline": "#39404b",
    },
}

FONT_STACK = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
FONT_IMPORT = ("https://fonts.googleapis.com/css2?"
               "family=IBM+Plex+Sans:wght@400;500;600;700&display=swap")


def theme_mode() -> str:
    try:
        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None) == "dark":
            return "dark"
    except Exception:
        pass
    return "light"


def is_dark() -> bool:
    return theme_mode() == "dark"


def chrome(mode: Optional[str] = None) -> dict:
    return _CHROME[mode or theme_mode()]


def categorical(mode: Optional[str] = None) -> list[str]:
    return CATEGORICAL_DARK if (mode or theme_mode()) == "dark" else CATEGORICAL_LIGHT


def status(mode: Optional[str] = None) -> dict:
    return STATUS_DARK if (mode or theme_mode()) == "dark" else STATUS


def accent(mode: Optional[str] = None) -> str:
    return TEAL_DARK if (mode or theme_mode()) == "dark" else TEAL


def series_primary(mode: Optional[str] = None) -> str:
    """The single-series data colour (mode-aware vivid teal)."""
    return SERIES_PRIMARY_DARK if (mode or theme_mode()) == "dark" else SERIES_PRIMARY


def rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' + alpha → 'rgba(r,g,b,a)' for translucent washes."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Plotly template ─────────────────────────────────────────────────────────────

def _build_template(mode: str) -> go.layout.Template:
    ch = _CHROME[mode]
    return go.layout.Template(
        layout=go.Layout(
            colorway=CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT,
            font=dict(family=FONT_STACK, color=ch["ink"], size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Roomy margins so the title clears the top and the x-axis title /
            # any below-plot legend clear the bottom — nothing sits flush against
            # the chart-card edge where it would look clipped. Right margin holds
            # outside value labels on horizontal bars.
            margin=dict(l=10, r=28, t=64, b=64),
            title=dict(font=dict(size=15, color=ch["ink"]),
                       x=0, xanchor="left", y=1.0, yanchor="top",
                       yref="container", pad=dict(t=2, b=14)),
            # Recessive chrome: thin gridlines, no y-baseline clutter, ticks muted.
            # automargin lets Plotly grow the margin to fit long category labels
            # and axis titles, so nothing is clipped by the fixed chart-card box.
            xaxis=dict(gridcolor=ch["grid"], gridwidth=1, linecolor=ch["baseline"],
                       zeroline=False, automargin=True,
                       tickfont=dict(color=ch["muted"], size=11),
                       title=dict(font=dict(color=ch["ink_secondary"], size=12),
                                  standoff=8)),
            yaxis=dict(gridcolor=ch["grid"], gridwidth=1, linecolor="rgba(0,0,0,0)",
                       zeroline=False, ticks="", automargin=True,
                       tickfont=dict(color=ch["muted"], size=11),
                       title=dict(font=dict(color=ch["ink_secondary"], size=12),
                                  standoff=8)),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                        xref="paper", yref="paper",
                        font=dict(color=ch["ink_secondary"], size=11),
                        bgcolor="rgba(0,0,0,0)"),
            # Designed tooltip: surface card, bordered, left-aligned — matches the
            # dashboard's card language instead of Plotly's default black/white.
            hoverlabel=dict(font=dict(family=FONT_STACK, size=12, color=ch["ink"]),
                            bgcolor=ch["surface"], bordercolor=ch["border"],
                            align="left"),
            modebar=dict(bgcolor="rgba(0,0,0,0)", color=ch["muted"],
                         activecolor=ch["ink_secondary"]),
            colorscale=dict(sequential=[[i / (len(SEQUENTIAL_TEAL) - 1), c]
                                        for i, c in enumerate(SEQUENTIAL_TEAL)]),
            bargap=0.24,
        ),
        # Trace-level defaults so EVERY chart inherits the mark spec — rounded,
        # baseline-anchored bar ends with a 2px surface spacer, and a thin surface
        # ring on markers so overlapping dots stay legible. Explicit per-trace
        # settings still win, so bespoke charts are unaffected.
        data=dict(
            bar=[go.Bar(marker=dict(cornerradius=4,
                                    line=dict(width=1.5, color=ch["surface"])))],
            histogram=[go.Histogram(marker=dict(
                line=dict(width=1, color=ch["surface"])))],
            scatter=[go.Scatter(marker=dict(
                line=dict(width=1.25, color=ch["surface"])))],
        ),
    )


pio.templates["sph_light"] = _build_template("light")
pio.templates["sph_dark"] = _build_template("dark")


def active_template() -> str:
    return "sph_dark" if is_dark() else "sph_light"


def apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=active_template())
    return fig


# ── Global CSS ──────────────────────────────────────────────────────────────────

def inject_css() -> None:
    ch = chrome()
    ac = accent()
    st.markdown(
        f"""
        <style>
        @import url('{FONT_IMPORT}');
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], [data-testid="stMetricValue"],
        button, input, select, textarea {{ font-family: {FONT_STACK}; }}

        /* ── Product shell: off-white page, white surfaces for depth ─────────── */
        .stApp, [data-testid="stAppViewContainer"] {{ background: {ch['page']}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stAppDeployButton"], .stDeployButton {{ display: none !important; }}
        .block-container {{ padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1360px; }}
        [data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}

        /* ── Sidebar — a designed shell, not default nav ─────────────────────── */
        section[data-testid="stSidebar"] {{ background: {ch['surface']};
            border-right: 1px solid {ch['border']}; }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
        .sph-brand {{ padding: 2px 2px 12px; margin-bottom: 6px;
            border-bottom: 1px solid {ch['border']}; }}
        .sph-logo {{ background: #fff; border-radius: 12px; padding: 12px 14px;
            border: 1px solid {ch['border']}; box-shadow: 0 1px 3px {rgba(ch['ink'], 0.06)}; }}
        .sph-logo img {{ width: 100%; display: block; }}
        .sph-brand-sub {{ font-size: 10.5px; font-weight: 700; letter-spacing: 0.15em;
            color: {ch['ink_secondary']}; text-transform: uppercase; text-align: center;
            margin-top: 9px; }}
        .sph-sidebar-gap {{ height: 14px; border-top: 1px solid {ch['border']};
            margin: 14px 2px 0; }}

        /* Sidebar nav — quiet links, hover, and an accent chip for the active page. */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
            border-radius: 9px; padding: 8px 12px; margin: 1px 0;
            color: {ch['ink_secondary']}; font-size: 13.5px; font-weight: 500;
            border-left: 2px solid transparent; transition: all .12s ease; }}
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
            background: {ch['surface_2']}; color: {ch['ink']}; }}
        .sph-nav-active {{ display: flex; align-items: center; gap: 11px;
            border-radius: 9px; padding: 9px 12px; margin: 1px 0;
            background: {rgba(ac, 0.13)}; color: {ac}; font-size: 13.5px;
            font-weight: 700; border-left: 2px solid var(--accent, {ac}); }}
        .sph-nav-ic {{ font-family: 'Material Symbols Rounded'; font-size: 20px;
            font-weight: normal; line-height: 1; color: {ac};
            font-variant-ligatures: normal; -webkit-font-feature-settings: 'liga'; }}
        /* inactive page-link icons: quiet, aligned with the active chip */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a span[data-testid="stIconMaterial"] {{
            font-size: 20px !important; margin-right: 3px; }}

        /* ── Page header: dominant title, readable question, subtle context ──── */
        .sph-header {{ border-bottom: 1px solid {ch['border']}; padding-bottom: 16px;
            margin-bottom: 22px; }}
        .sph-title {{ font-family: {FONT_STACK}; font-size: 31px; font-weight: 700;
            letter-spacing: -0.02em; color: {ch['ink']}; margin: 0 0 6px; line-height: 1.1; }}
        .sph-sub {{ font-size: 15.5px; font-weight: 500; color: {ch['ink_secondary']};
            margin: 0; line-height: 1.45; max-width: 74ch; }}
        .sph-meta {{ font-size: 11.5px; color: {ch['muted']}; margin-top: 10px;
            letter-spacing: 0.02em; text-transform: none; }}

        /* ── Section header (chapter) + optional analytical finding ──────────── */
        .sph-sec {{ margin: 40px 0 14px; padding-bottom: 9px;
            border-bottom: 1px solid {ch['border']}; }}
        .sph-sec-title {{ font-size: 16.5px; font-weight: 700; letter-spacing: -0.012em;
            color: {ch['ink']}; line-height: 1.25;
            border-left: 3px solid var(--accent, {ac}); padding-left: 11px; }}
        .sph-sec-find {{ font-size: 14px; font-weight: 700; color: var(--accent, {ac});
            margin: 5px 0 0 14px; letter-spacing: -0.003em; }}
        /* legacy alias + Streamlit subheader kept visually identical */
        .sph-section {{ font-size: 16.5px; font-weight: 700; color: {ch['ink']};
            margin: 40px 0 14px; padding: 0 0 9px 11px; letter-spacing: -0.012em;
            border-left: 3px solid var(--accent, {ac});
            border-bottom: 1px solid {ch['border']}; }}
        h3 {{ font-size: 16.5px !important; font-weight: 700 !important;
            letter-spacing: -0.012em !important; color: {ch['ink']} !important;
            padding: 0 0 9px 11px !important; border-left: 3px solid var(--accent, {ac}) !important;
            border-bottom: 1px solid {ch['border']} !important; margin: 40px 0 14px !important; }}

        /* ── KPI scorecard: number dominant, label small, qualifier readable ─── */
        .sph-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(184px, 1fr));
            gap: 15px; margin: 6px 0 12px; }}
        .sph-tile {{ position: relative; background: {ch['surface']};
            border: 1px solid {ch['border']}; border-radius: 14px;
            padding: 18px 18px 16px; box-shadow: 0 1px 2px rgba(20,24,31,0.05);
            transition: box-shadow .16s ease, transform .16s ease, border-color .16s ease; }}
        .sph-tile::before {{ content: ""; position: absolute; left: 16px; right: 16px; top: 0;
            height: 2px; border-radius: 2px; background: var(--accent, {ac}); opacity: .55; }}
        .sph-tile:hover {{ box-shadow: 0 8px 22px rgba(20,24,31,0.10);
            transform: translateY(-2px); border-color: {rgba(ac, 0.35)}; }}
        .sph-tile-label {{ font-size: 10.5px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: {ch['muted']}; margin-bottom: 10px; line-height: 1.35; }}
        .sph-tile-value {{ font-size: 33px; font-weight: 700; letter-spacing: -0.025em;
            color: {ch['ink']}; line-height: 1.02; word-break: break-word;
            font-variant-numeric: tabular-nums; }}
        .sph-tile-detail {{ font-size: 12px; color: {ch['ink_secondary']}; margin-top: 9px;
            line-height: 1.45; }}

        .sph-badge {{ display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 9px;
            border-radius: 999px; color: #fff; letter-spacing: 0.02em; }}

        /* Expander summary reads as a sub-section header — never faint. */
        [data-testid="stExpander"] {{ border: 1px solid {ch['border']} !important;
            border-radius: 12px !important; background: {ch['surface']}; }}
        [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {{
            font-weight: 600 !important; color: {ch['ink']} !important; font-size: 14px !important; }}

        /* Charts render frameless so nothing is clipped by a border. */
        [data-testid="stPlotlyChart"] {{ overflow: visible; }}
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot {{ width: 100% !important; }}

        /* Tables: analytical, not raw — light frame, lining figures, quiet header. */
        [data-testid="stDataFrame"] {{ border: 1px solid {ch['border']};
            border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(20,24,31,0.04);
            font-variant-numeric: tabular-nums; }}
        [data-testid="stDataFrame"] [role="columnheader"] {{
            text-transform: uppercase; letter-spacing: 0.04em; font-size: 11px;
            color: {ch['muted']}; }}

        /* Captions/qualifiers: readable, never faint (caveats must be legible). */
        [data-testid="stCaptionContainer"], .stCaption {{ color: {ch['ink_secondary']};
            font-size: 12.5px; line-height: 1.55; max-width: 90ch; }}

        /* Info / empty states in the brand surface tone. */
        [data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid {ch['border']};
            background: {ch['surface_2']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Formatters ────────────────────────────────────────────────────────────────

def _is_nan(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def fmt_kes(value, decimals: int = 0) -> str:
    if _is_nan(value):
        return "—"
    return f"KES {value:,.{decimals}f}"


def fmt_kes_compact(value) -> str:
    if _is_nan(value):
        return "—"
    m = abs(value)
    if m >= 1_000_000:
        return f"KES {value / 1_000_000:,.1f}M"
    if m >= 10_000:
        return f"KES {value / 1_000:,.0f}K"
    return f"KES {value:,.0f}"


def fmt_compact(value) -> str:
    if _is_nan(value):
        return "—"
    m = abs(value)
    if m >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if m >= 10_000:
        return f"{value / 1_000:,.0f}K"
    if m >= 100:
        return f"{value:,.0f}"
    return f"{value:,.1f}" if isinstance(value, float) and value != int(value) else f"{value:,.0f}"


def fmt_pct(value, decimals: int = 1) -> str:
    if _is_nan(value):
        return "—"
    return f"{value * 100:,.{decimals}f}%"


def fmt_days(value, decimals: int = 0) -> str:
    if _is_nan(value):
        return "—"
    return f"{value:,.{decimals}f} d"


def fmt_date(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return value.strftime("%d %b %Y")
    except AttributeError:
        return str(value)[:10]
