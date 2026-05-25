"""
charts.py — Afya Clinical Analytics
=====================================
Reusable Plotly chart wrappers.
Every chart function returns a go.Figure ready for st.plotly_chart().
All charts use the Afya template from ui_template.py.

Chart catalogue:
  line_chart          — time-series line, multi-series optional
  bar_chart           — vertical bar, single or grouped
  hbar_chart          — horizontal bar (rankings, top-N lists)
  stacked_bar         — stacked vertical bar
  stacked_area        — area chart for burden group trends
  funnel_chart        — conversion funnel (inpatient pathway)
  heatmap             — hour × day demand heatmap
  scatter             — two-metric scatter with optional size
  donut               — proportion / share chart
  table_fig           — styled Plotly table
  waterfall           — revenue at risk / change breakdown
  bullet              — KPI vs benchmark (retention rate etc.)
  sparkline           — inline mini line for patient card vitals
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

from ksh.clinical_module.ui_template import (
    AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY,
    CHART_LAYOUT, AXIS, SEQ, BG_LIGHT, BORDER,
)

# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _ax(**overrides) -> dict:
    """Merge AXIS defaults with per-call overrides."""
    return {**AXIS, **overrides}


def _layout(**overrides) -> dict:
    """Merge CHART_LAYOUT defaults with per-call overrides."""
    return {**CHART_LAYOUT, **overrides}


def _rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _empty_fig(msg: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=GRAY, family="Montserrat"),
    )
    fig.update_layout(**_layout(height=200))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _safe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return None if df is empty or None."""
    if df is None or df.empty:
        return None
    return df


# ─── LINE CHART ───────────────────────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    *,
    color_map: dict | None = None,
    title: str = "",
    y_label: str = "",
    x_label: str = "",
    height: int = 320,
    show_markers: bool = True,
    y_format: str = "",        # "KES", "pct", "" for raw
    spike: bool = False,       # highlight spike/dip months
    spike_col: str = "",       # column name with 1/0 spike flag
    dip_col: str = "",
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    ys = [y] if isinstance(y, str) else y
    colors = color_map or {s: SEQ[i % len(SEQ)] for i, s in enumerate(ys)}

    fig = go.Figure()

    for series in ys:
        if series not in df.columns:
            continue
        col = colors.get(series, AFYA_BLUE)
        fig.add_trace(go.Scatter(
            x=df[x], y=df[series],
            name=series,
            mode="lines+markers" if show_markers else "lines",
            line=dict(color=col, width=2),
            marker=dict(color=col, size=5),
            hovertemplate=f"<b>{series}</b><br>%{{x}}<br>%{{y}}<extra></extra>",
        ))

    # Spike / dip annotations
    if spike and spike_col and spike_col in df.columns:
        spikes = df[df[spike_col] == 1]
        for _, row in spikes.iterrows():
            fig.add_vline(
                x=row[x], line_dash="dot",
                line_color=_rgba(ORANGE, 0.5), line_width=1,
            )

    if spike and dip_col and dip_col in df.columns:
        dips = df[df[dip_col] == 1]
        for _, row in dips.iterrows():
            fig.add_vline(
                x=row[x], line_dash="dot",
                line_color=_rgba(CORAL, 0.5), line_width=1,
            )

    fig.update_layout(
        **_layout(height=height),
        xaxis=_ax(title=x_label),
        yaxis=_ax(title=y_label, tickformat=_tick_fmt(y_format)),
    )
    return fig


# ─── BAR CHART ────────────────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    *,
    color_map: dict | None = None,
    title: str = "",
    y_label: str = "",
    x_label: str = "",
    height: int = 320,
    barmode: str = "group",    # "group" or "stack"
    y_format: str = "",
    show_text: bool = False,
    color: str = AFYA_BLUE,
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    ys = [y] if isinstance(y, str) else y
    colors = color_map or {s: SEQ[i % len(SEQ)] for i, s in enumerate(ys)}

    fig = go.Figure()
    for series in ys:
        if series not in df.columns:
            continue
        col = colors.get(series, color)
        fig.add_trace(go.Bar(
            x=df[x], y=df[series],
            name=series,
            marker_color=col,
            text=df[series].apply(lambda v: _fmt_val(v, y_format)) if show_text else None,
            textposition="outside" if show_text else "none",
            hovertemplate=f"<b>{series}</b><br>%{{x}}<br>%{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **_layout(height=height),
        barmode=barmode,
        xaxis=_ax(title=x_label),
        yaxis=_ax(title=y_label, tickformat=_tick_fmt(y_format)),
    )
    return fig


# ─── HORIZONTAL BAR ───────────────────────────────────────────────────────────

def hbar_chart(
    df: pd.DataFrame,
    x: str,          # numeric column
    y: str,          # category column
    *,
    color: str = AFYA_BLUE,
    color_col: str = "",       # column to drive colour scale
    title: str = "",
    x_label: str = "",
    height: int = 320,
    y_format: str = "",
    top_n: int = 0,            # 0 = show all
    show_text: bool = True,
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    plot_df = df.copy()
    if top_n > 0:
        plot_df = plot_df.nlargest(top_n, x)
    plot_df = plot_df.sort_values(x, ascending=True)

    if color_col and color_col in plot_df.columns:
        bar_colors = px.colors.sample_colorscale(
            "Blues",
            (plot_df[color_col] - plot_df[color_col].min())
            / max(plot_df[color_col].max() - plot_df[color_col].min(), 1),
        )
    else:
        bar_colors = color

    fig = go.Figure(go.Bar(
        x=plot_df[x],
        y=plot_df[y],
        orientation="h",
        marker_color=bar_colors,
        text=plot_df[x].apply(lambda v: _fmt_val(v, y_format)) if show_text else None,
        textposition="outside" if show_text else "none",
        hovertemplate=f"<b>%{{y}}</b><br>{x}: %{{x}}<extra></extra>",
    ))

    fig.update_layout(
        **_layout(height=max(height, len(plot_df) * 28 + 40)),
        xaxis=_ax(title=x_label, tickformat=_tick_fmt(y_format)),
        yaxis=_ax(automargin=True),
    )
    return fig


# ─── STACKED BAR ──────────────────────────────────────────────────────────────

def stacked_bar(
    df: pd.DataFrame,
    x: str,
    categories: list[str],
    *,
    color_map: dict | None = None,
    y_label: str = "",
    x_label: str = "",
    height: int = 320,
    y_format: str = "",
    normalized: bool = False,  # 100% stacked
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    colors = color_map or {s: SEQ[i % len(SEQ)] for i, s in enumerate(categories)}
    barmode = "relative" if normalized else "stack"

    fig = go.Figure()
    for cat in categories:
        if cat not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=df[x], y=df[cat],
            name=cat,
            marker_color=colors.get(cat, GRAY),
            hovertemplate=f"<b>{cat}</b><br>%{{x}}<br>%{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **_layout(height=height),
        barmode=barmode,
        xaxis=_ax(title=x_label),
        yaxis=_ax(
            title=y_label,
            tickformat="%" if normalized else _tick_fmt(y_format),
        ),
    )
    return fig


# ─── STACKED AREA ─────────────────────────────────────────────────────────────

def stacked_area(
    df: pd.DataFrame,
    x: str,
    categories: list[str],
    *,
    color_map: dict | None = None,
    y_label: str = "",
    x_label: str = "",
    height: int = 320,
    y_format: str = "",
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    colors = color_map or {s: SEQ[i % len(SEQ)] for i, s in enumerate(categories)}

    fig = go.Figure()
    for cat in categories:
        if cat not in df.columns:
            continue
        col = colors.get(cat, GRAY)
        fig.add_trace(go.Scatter(
            x=df[x], y=df[cat],
            name=cat,
            stackgroup="one",
            mode="lines",
            line=dict(color=col, width=1),
            fillcolor=_rgba(col, 0.5),
            hovertemplate=f"<b>{cat}</b><br>%{{x}}<br>%{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **_layout(height=height),
        xaxis=_ax(title=x_label),
        yaxis=_ax(title=y_label, tickformat=_tick_fmt(y_format)),
    )
    return fig


# ─── FUNNEL CHART ─────────────────────────────────────────────────────────────

def funnel_chart(
    labels: list[str],
    values: list[float],
    *,
    conversion_rates: list[float] | None = None,
    title: str = "Inpatient Conversion Funnel",
    height: int = 380,
    caveat: str = "",
) -> go.Figure:
    """
    Conversion funnel for the inpatient pathway.
    labels:           gate names (e.g. ["Visit created", "Saw a doctor", ...])
    values:           count at each gate
    conversion_rates: optional pct at each gate vs prior gate
    caveat:           note to show as annotation (Gate 2 note recording caveat)
    """
    if not labels or not values:
        return _empty_fig()

    bar_colors = [
        _rgba(AFYA_BLUE, 1 - i * 0.15)
        for i in range(len(labels))
    ]

    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent initial",
        marker=dict(color=bar_colors),
        connector=dict(line=dict(color=BORDER, width=1)),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
    ))

    if caveat:
        fig.add_annotation(
            text=f"⚠ {caveat}",
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=10, color=ORANGE, family="Montserrat"),
            bgcolor=_rgba(ORANGE, 0.08),
            bordercolor=ORANGE,
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        **_layout(height=height),
        funnelmode="stack",
        yaxis=_ax(),
    )
    return fig


# ─── HEATMAP ──────────────────────────────────────────────────────────────────

def heatmap(
    df: pd.DataFrame,
    x: str,           # hour
    y: str,           # day name
    z: str,           # visit count
    *,
    x_label: str = "Hour of Day (EAT)",
    y_label: str = "Day of Week",
    height: int = 300,
    colorscale: str = "Blues",
    day_order: list[str] | None = None,
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="sum", fill_value=0)

    if day_order:
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale=colorscale,
        hovertemplate="<b>%{y} %{x}:00</b><br>Visits: %{z:,}<extra></extra>",
        colorbar=dict(
            thickness=12,
            tickfont=dict(family="Montserrat", size=9, color=AFYA_BLUE),
        ),
    ))

    fig.update_layout(
        **_layout(height=height),
        xaxis=_ax(title=x_label),
        yaxis=_ax(title=y_label, autorange="reversed"),
    )
    return fig


# ─── SCATTER ──────────────────────────────────────────────────────────────────

def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color_col: str = "",
    size_col: str = "",
    label_col: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = 340,
    x_format: str = "",
    y_format: str = "",
    color_map: dict | None = None,
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    plot_df = df.copy()

    # Normalise size
    sizes = None
    if size_col and size_col in plot_df.columns:
        mn, mx = plot_df[size_col].min(), plot_df[size_col].max()
        rng = max(mx - mn, 1)
        sizes = ((plot_df[size_col] - mn) / rng * 24 + 6).tolist()

    colors = AFYA_BLUE
    if color_col and color_col in plot_df.columns:
        unique_vals = plot_df[color_col].unique()
        if color_map:
            colors = [color_map.get(str(v), GRAY) for v in plot_df[color_col]]
        else:
            palette = {v: SEQ[i % len(SEQ)] for i, v in enumerate(unique_vals)}
            colors = [palette[v] for v in plot_df[color_col]]

    fig = go.Figure(go.Scatter(
        x=plot_df[x], y=plot_df[y],
        mode="markers+text" if label_col else "markers",
        text=plot_df[label_col].tolist() if label_col and label_col in plot_df.columns else None,
        textposition="top center",
        textfont=dict(size=9, family="Montserrat", color=COOL_BLUE),
        marker=dict(
            color=colors,
            size=sizes or 8,
            opacity=0.8,
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            f"<b>%{{text}}</b><br>" if label_col else ""
        ) + f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
    ))

    fig.update_layout(
        **_layout(height=height),
        xaxis=_ax(title=x_label, tickformat=_tick_fmt(x_format)),
        yaxis=_ax(title=y_label, tickformat=_tick_fmt(y_format)),
    )
    return fig


# ─── DONUT ────────────────────────────────────────────────────────────────────

def donut(
    labels: list[str],
    values: list[float],
    *,
    color_map: dict | None = None,
    title: str = "",
    height: int = 280,
    hole: float = 0.55,
    center_label: str = "",
    center_value: str = "",
) -> go.Figure:
    if not labels or not values:
        return _empty_fig()

    colors = [
        (color_map or {}).get(str(l), SEQ[i % len(SEQ)])
        for i, l in enumerate(labels)
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Montserrat", size=10, color=COOL_BLUE),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
        sort=False,
    ))

    if center_label or center_value:
        fig.add_annotation(
            text=f"<b>{center_value}</b><br><span style='font-size:9px'>{center_label}</span>",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=AFYA_BLUE, family="Montserrat"),
            xref="paper", yref="paper",
        )

    fig.update_layout(**_layout(height=height))
    return fig


# ─── PLOTLY TABLE ─────────────────────────────────────────────────────────────

def table_fig(
    df: pd.DataFrame,
    *,
    col_labels: dict | None = None,     # {col_name: display_label}
    col_widths: list[int] | None = None,
    height: int | None = None,
    header_color: str = COOL_BLUE,
    highlight_col: str = "",            # column to colour-code
    highlight_fn=None,                  # fn(val) → CSS colour string
    fmt: dict | None = None,            # {col: "KES" | "pct" | "num"}
) -> go.Figure:
    if _safe(df) is None:
        return _empty_fig()

    display_df = df.copy()

    # Apply formatting
    if fmt:
        for col, style in fmt.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: _fmt_val(v, style)
                )

    # Column headers
    headers = [
        (col_labels or {}).get(c, c.replace("_", " ").title())
        for c in display_df.columns
    ]

    # Row fill colours
    n_rows = len(display_df)
    row_fill = [[BG_LIGHT if i % 2 == 0 else "#FFFFFF" for i in range(n_rows)]]

    fig = go.Figure(go.Table(
        columnwidth=col_widths,
        header=dict(
            values=headers,
            fill_color=header_color,
            font=dict(color="white", size=10, family="Montserrat"),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[display_df[c].tolist() for c in display_df.columns],
            fill_color=row_fill * len(display_df.columns),
            font=dict(color=COOL_BLUE, size=10, family="Montserrat"),
            align="left",
            height=24,
        ),
    ))

    layout_args = dict(margin=dict(l=0, r=0, t=0, b=0))
    if height:
        layout_args["height"] = height

    fig.update_layout(**layout_args)
    return fig


# ─── WATERFALL ────────────────────────────────────────────────────────────────

def waterfall(
    labels: list[str],
    values: list[float],
    *,
    measure: list[str] | None = None,   # "relative", "total", "absolute"
    y_label: str = "KES",
    height: int = 340,
    y_format: str = "KES",
) -> go.Figure:
    if not labels or not values:
        return _empty_fig()

    measures = measure or (
        ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]
    )

    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        textposition="outside",
        text=[_fmt_val(v, y_format) for v in values],
        connector=dict(line=dict(color=BORDER, width=1)),
        increasing=dict(marker=dict(color=TEAL)),
        decreasing=dict(marker=dict(color=CORAL)),
        totals=dict(marker=dict(color=AFYA_BLUE)),
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
    ))

    fig.update_layout(
        **_layout(height=height),
        yaxis=_ax(title=y_label, tickformat=_tick_fmt(y_format)),
        showlegend=False,
    )
    return fig


# ─── BULLET (KPI VS BENCHMARK) ────────────────────────────────────────────────

def bullet(
    actual: float,
    benchmark: float,
    label: str,
    *,
    low_threshold: float | None = None,
    format: str = "pct",
    height: int = 80,
) -> go.Figure:
    """Single-row bullet chart: actual value vs benchmark line."""
    color = TEAL if actual >= benchmark else CORAL

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=actual,
        delta=dict(
            reference=benchmark,
            valueformat=".1f",
            increasing=dict(color=TEAL),
            decreasing=dict(color=CORAL),
        ),
        title=dict(text=label, font=dict(family="Montserrat", size=11, color=AFYA_BLUE)),
        number=dict(
            suffix="%" if format == "pct" else "",
            font=dict(family="Montserrat", size=20, color=color),
        ),
        gauge=dict(
            axis=dict(range=[0, max(actual, benchmark) * 1.2], visible=False),
            bar=dict(color=color, thickness=0.4),
            threshold=dict(
                line=dict(color=AFYA_BLUE, width=2),
                thickness=0.75,
                value=benchmark,
            ),
            steps=[
                dict(range=[0, benchmark], color=_rgba(CORAL, 0.08)),
                dict(range=[benchmark, max(actual, benchmark) * 1.2],
                     color=_rgba(TEAL, 0.08)),
            ],
        ),
    ))

    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="#fff")
    return fig


# ─── SPARKLINE ────────────────────────────────────────────────────────────────

def sparkline(
    values: list[float],
    *,
    color: str = AFYA_BLUE,
    trend: str = "Stable",     # "Improving", "Worsening", "Stable", "Insufficient data"
    height: int = 60,
) -> go.Figure:
    """Inline mini line chart for patient card vitals."""
    if not values or len(values) < 2:
        return _empty_fig("Insufficient data")

    trend_color = {
        "Improving": TEAL,
        "Worsening": CORAL,
        "Stable": AFYA_BLUE,
        "Insufficient data": GRAY,
    }.get(trend, AFYA_BLUE)

    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines+markers",
        line=dict(color=trend_color, width=2),
        marker=dict(color=trend_color, size=4),
        hovertemplate="%{y}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ─── TICK FORMAT HELPER ───────────────────────────────────────────────────────

def _tick_fmt(style: str) -> str:
    if style == "KES":
        return ",.0f"
    if style == "pct":
        return ".1f"
    if style == "K":
        return ",.1s"
    return ""


def _fmt_val(v, style: str) -> str:
    """Format a value for text labels on charts."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if style == "KES":
        if abs(v) >= 1_000_000:
            return f"KES {v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"KES {v / 1_000:.0f}K"
        return f"KES {v:.0f}"
    if style == "pct":
        return f"{v:.1f}%"
    if style == "num":
        return f"{v:,.0f}"
    return f"{v:,.1f}"


# ─── BURDEN GROUP COLOUR MAP (shared across all tabs) ────────────────────────

BURDEN_COLORS = {
    "Communicable":         TEAL,
    "NCD / Chronic":        AFYA_BLUE,
    "RMNCH - Maternal":     PURPLE,
    "RMNCH - Perinatal":    _rgba(PURPLE, 0.6),
    "RMNCH":                PURPLE,
    "Injury":               ORANGE,
    "Mental Health":        CORAL,
    "Oncology":             COOL_BLUE,
    "Respiratory":          _rgba(TEAL, 0.7),
    "NCD - Cardiovascular": _rgba(AFYA_BLUE, 0.7),
    "NCD - Endocrine / Metabolic": _rgba(AFYA_BLUE, 0.85),
    "NCD - Neurologic":     _rgba(AFYA_BLUE, 0.6),
    "NCD - Other":          _rgba(AFYA_BLUE, 0.5),
    "Health Service Encounter": GRAY,
    "Signs & Symptoms - Unclassified": _rgba(GRAY, 0.6),
    "Unclassified":         _rgba(GRAY, 0.4),
    "Unknown - No diagnosis recorded": _rgba(GRAY, 0.3),
}

# Lifecycle / retention colours
LIFECYCLE_COLORS = {
    "1. Active (≤90 days)":   TEAL,
    "2. Lapsing (91–180 days)": ORANGE,
    "3. LTFU (>180 days)":    CORAL,
    "N/A — surgical":         GRAY,
    "N/A — acute":            _rgba(GRAY, 0.6),
    "N/A — self_discharge":   _rgba(CORAL, 0.4),
    "N/A — unclassified":     _rgba(GRAY, 0.3),
    "New patient":            _rgba(TEAL, 0.5),
    "Active":                 TEAL,
    "Lapsing":                ORANGE,
    "LTFU — returning":       AFYA_BLUE,
}

# Risk badge colours
BADGE_COLORS = {
    "HIGH":  CORAL,
    "WATCH": ORANGE,
    "OK":    TEAL,
}