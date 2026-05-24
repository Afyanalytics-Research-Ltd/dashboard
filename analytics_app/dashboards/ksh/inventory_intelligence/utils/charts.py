"""
Plotly chart builders — all use the Afya design-system palette.
Every chart is transparent (paper + plot bgcolor = rgba(0,0,0,0)) for
seamless embedding in white cards.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.formatting import (
    COLOR_AMBER,
    COLOR_PRIMARY,
    COLOR_RED,
    DOS_COLORS,
    STATUS_COLORS,
    fmt_kes_millions,
)

_LAYOUT = dict(
    font_family="sans-serif",
    font_color="#1A1A2E",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=32, b=0),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font_size=11,
    ),
)

_GRIDLINE = dict(color="#E5E7EB", width=1)


# ── Inventory capital waterfall ────────────────────────────────────────────────

def inventory_waterfall(
    healthy: float,
    slow_moving: float,
    dead: float,
    near_expiry: float,
    stockout: float,
) -> go.Figure:
    """Stacked horizontal bar showing inventory capital composition."""
    total = healthy + slow_moving + dead + near_expiry + stockout or 1
    categories = ["Healthy", "Slow moving (30–90d)", "Dead stock (90d+)", "Near expiry", "Stocked out"]
    values = [healthy, slow_moving, dead, near_expiry, stockout]
    colors = [COLOR_PRIMARY, "#1D9E75", COLOR_AMBER, COLOR_RED, "#791F1F"]

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        pct = val / total * 100
        fig.add_trace(go.Bar(
            name=f"{cat} ({pct:.1f}%)",
            x=[val], y=["Inventory"],
            orientation="h",
            marker_color=col,
            text=fmt_kes_millions(val),
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{cat}</b><br>{fmt_kes_millions(val)} ({pct:.1f}%)<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        barmode="stack",
        height=90,
        showlegend=True,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


# ── Days-of-stock bar chart ───────────────────────────────────────────────────

def dos_bar_chart(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Horizontal bar: days of stock remaining.
    Accepts uppercase or lowercase column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    dos_col = "days_of_stock_p50" if "days_of_stock_p50" in df.columns else "days_of_stock"
    df[dos_col] = pd.to_numeric(df[dos_col], errors="coerce")
    df = df.dropna(subset=[dos_col]).nsmallest(top_n, dos_col).sort_values(dos_col)
    colors = df["dos_status"].map(DOS_COLORS).fillna("#888780")

    fig = go.Figure(go.Bar(
        x=df[dos_col],
        y=df["canonical_name"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Days of stock: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        height=max(300, top_n * 24),
        xaxis=dict(title="Days of stock remaining", gridcolor=_GRIDLINE["color"]),
        yaxis=dict(tickfont=dict(size=10)),
    )
    fig.add_vline(x=7,  line_dash="dash", line_color=COLOR_RED,   annotation_text="7d critical",  annotation_position="top")
    fig.add_vline(x=30, line_dash="dash", line_color=COLOR_AMBER, annotation_text="30d low", annotation_position="top")
    return fig


# ── Stock status donut ────────────────────────────────────────────────────────

# Human-readable display labels for each status key
_DONUT_LABELS: dict[str, str] = {
    "stockout":  "Stocked out",
    "zero":      "Stocked out",
    "negative":  "Negative SOH",
    "critical":  "Critical  < 7d",
    "low":       "Low  7–30d",
    "adequate":  "Adequate",
}

# Distinct colour per severity tier — deliberately wider range than STATUS_COLORS
_DONUT_COLORS: dict[str, str] = {
    "negative":  "#7F1D1D",   # darkest red
    "stockout":  "#991B1B",   # dark red
    "zero":      "#991B1B",
    "critical":  "#DC2626",   # bright red
    "low":       "#D97706",   # amber
    "adequate":  "#0F6E56",   # teal
}

# Canonical ordering (worst → best) so segments render predictably
_DONUT_ORDER = ["negative", "stockout", "zero", "critical", "low", "adequate"]


def status_donut(status_counts: dict) -> go.Figure:
    """
    Donut chart of stock status distribution.
    Accepts any subset of status keys; segments always render worst-first.
    """
    ordered_keys = [k for k in _DONUT_ORDER if k in status_counts]
    # Include any unexpected keys at the end
    ordered_keys += [k for k in status_counts if k not in _DONUT_ORDER]

    labels = [_DONUT_LABELS.get(k, k.title()) for k in ordered_keys]
    values = [status_counts[k] for k in ordered_keys]
    colors = [_DONUT_COLORS.get(k, "#888780") for k in ordered_keys]

    total = sum(values)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.66,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,} products (%{percent})<extra></extra>",
        textinfo="none",
        sort=False,
    ))

    fig.add_annotation(
        text=(
            f"<b style='font-size:22px;color:#111827'>{total:,}</b>"
            f"<br><span style='font-size:11px;color:#9CA3AF'>products</span>"
        ),
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(family="sans-serif"),
    )

    # _LAYOUT already contains a 'legend' key — override it without duplication
    layout = {k: v for k, v in _LAYOUT.items() if k not in ("legend", "margin")}
    fig.update_layout(
        **layout,
        height=280,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.03,
            font=dict(size=11, color="#6B7280"),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=0, r=110, t=8, b=8),
    )
    return fig


# ── Monthly dispensing trend ──────────────────────────────────────────────────

def dispensing_trend(
    df: pd.DataFrame,
    y_col: str = "TOTAL_UNITS_DISPENSED",
    color_col: Optional[str] = "CANONICAL_NAME",
    title: str = "",
) -> go.Figure:
    """Line chart of monthly dispensing with markers."""
    fig = px.line(
        df,
        x="DISPENSING_MONTH",
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        markers=True,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        **_LAYOUT,
        height=340,
        xaxis=dict(gridcolor=_GRIDLINE["color"]),
        yaxis=dict(gridcolor=_GRIDLINE["color"]),
    )
    fig.update_traces(line_width=2)
    return fig


# ── Demand forecast chart ─────────────────────────────────────────────────────

def demand_forecast_chart(
    monthly_df: pd.DataFrame,
    avg_daily: float,
    ci_lower_30d: float,
    ci_upper_30d: float,
    product_name: str = "",
) -> go.Figure:
    """
    Historical monthly bars + 30-day forecast point + CI band.
    monthly_df: DISPENSING_MONTH, TOTAL_UNITS_DISPENSED
    """
    fig = go.Figure()

    if not monthly_df.empty:
        fig.add_trace(go.Bar(
            x=monthly_df["DISPENSING_MONTH"],
            y=monthly_df["TOTAL_UNITS_DISPENSED"],
            name="Historical",
            marker_color=COLOR_PRIMARY,
            opacity=0.7,
        ))

    # Forecast point
    last_month = pd.to_datetime(monthly_df["DISPENSING_MONTH"].max()) if not monthly_df.empty else pd.Timestamp.now()
    next_month = last_month + pd.offsets.MonthBegin(1)
    forecast_val = avg_daily * 30

    fig.add_trace(go.Scatter(
        x=[next_month],
        y=[forecast_val],
        mode="markers",
        marker=dict(size=12, color=COLOR_AMBER, symbol="diamond"),
        name="30d forecast",
        hovertemplate=f"Forecast: {forecast_val:.0f} units<extra></extra>",
    ))

    # CI band
    fig.add_trace(go.Scatter(
        x=[next_month, next_month],
        y=[ci_lower_30d / 30 * 30, ci_upper_30d / 30 * 30],
        mode="lines",
        line=dict(color=COLOR_AMBER, width=0),
        showlegend=False,
    ))

    fig.add_annotation(
        x=next_month,
        y=forecast_val,
        text=f"  {forecast_val:.0f}u<br>  ({ci_lower_30d:.0f}–{ci_upper_30d:.0f})",
        showarrow=False,
        font_size=10,
        xanchor="left",
    )

    fig.update_layout(
        **_LAYOUT,
        height=300,
        title=f"{product_name} — Monthly consumption + 30d forecast" if product_name else "Monthly consumption",
        xaxis=dict(gridcolor=_GRIDLINE["color"]),
        yaxis=dict(title="Units", gridcolor=_GRIDLINE["color"]),
    )
    return fig


# ── Dead stock scatter ────────────────────────────────────────────────────────

def dead_stock_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter: days idle (x) vs historical dispensing value (y).
    Expects: CANONICAL_NAME, DAYS_IDLE, TOTAL_HISTORICAL_VALUE, THERAPEUTIC_CLASS
    """
    df = df.copy()
    df["DAYS_IDLE"]              = pd.to_numeric(df["DAYS_IDLE"],              errors="coerce")
    df["TOTAL_HISTORICAL_VALUE"] = pd.to_numeric(df["TOTAL_HISTORICAL_VALUE"], errors="coerce")
    fig = px.scatter(
        df,
        x="DAYS_IDLE",
        y="TOTAL_HISTORICAL_VALUE",
        color="THERAPEUTIC_CLASS",
        hover_name="CANONICAL_NAME",
        hover_data={"DAYS_IDLE": True, "TOTAL_HISTORICAL_VALUE": ":,.0f"},
        labels={"DAYS_IDLE": "Days since last dispense", "TOTAL_HISTORICAL_VALUE": "KES (historical value)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker_size=9, opacity=0.8)
    fig.update_layout(
        **_LAYOUT,
        height=400,
        xaxis=dict(gridcolor=_GRIDLINE["color"]),
        yaxis=dict(gridcolor=_GRIDLINE["color"]),
    )
    fig.add_vline(x=30, line_dash="dash", line_color=COLOR_AMBER, annotation_text="30d slow")
    fig.add_vline(x=90, line_dash="dash", line_color=COLOR_RED,   annotation_text="90d dead")
    return fig


# ── ABC pareto chart ──────────────────────────────────────────────────────────

def abc_pareto(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of products by dispensing value with cumulative % line.
    Expects: CANONICAL_NAME, TOTAL_DISPENSING_VALUE, ABC_CLASS
    """
    df = df.copy()
    df["TOTAL_DISPENSING_VALUE"] = pd.to_numeric(df["TOTAL_DISPENSING_VALUE"], errors="coerce").fillna(0)
    class_colors = {"A": COLOR_RED, "B": COLOR_AMBER, "C": COLOR_PRIMARY}
    df = df.sort_values("TOTAL_DISPENSING_VALUE", ascending=False).head(40)
    df["cumulative_pct"] = df["TOTAL_DISPENSING_VALUE"].cumsum() / df["TOTAL_DISPENSING_VALUE"].sum() * 100
    bar_colors = df["ABC_CLASS"].map(class_colors).fillna("#888780") if "ABC_CLASS" in df.columns else COLOR_PRIMARY

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["CANONICAL_NAME"],
        y=df["TOTAL_DISPENSING_VALUE"],
        name="Dispensing value",
        marker_color=bar_colors,
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Value: KES %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["CANONICAL_NAME"],
        y=df["cumulative_pct"],
        name="Cumulative %",
        line=dict(color="#1A1A2E", width=2),
        yaxis="y2",
    ))
    fig.add_hline(y=70, line_dash="dot", line_color=COLOR_RED,   yref="y2", annotation_text="A/B", annotation_position="right")
    fig.add_hline(y=90, line_dash="dot", line_color=COLOR_AMBER, yref="y2", annotation_text="B/C", annotation_position="right")

    fig.update_layout(
        **_LAYOUT,
        height=380,
        yaxis=dict(title="KES value", gridcolor=_GRIDLINE["color"]),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        bargap=0.15,
    )
    return fig


# ── Stockout timeline (Gantt-style) ───────────────────────────────────────────

def stockout_timeline(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal timeline of stockout episodes per product.
    Expects: CANONICAL_NAME, FIRST_STOCKOUT_AT, LAST_STOCKOUT_AT
    """
    df = df.head(top_n).copy()
    df["FIRST_STOCKOUT_AT"] = pd.to_datetime(df["FIRST_STOCKOUT_AT"])
    df["LAST_STOCKOUT_AT"] = pd.to_datetime(df["LAST_STOCKOUT_AT"])

    fig = go.Figure()
    for _, row in df.iterrows():
        duration = (row["LAST_STOCKOUT_AT"] - row["FIRST_STOCKOUT_AT"]).days + 1
        fig.add_trace(go.Bar(
            name=row["CANONICAL_NAME"],
            x=[duration],
            y=[row["CANONICAL_NAME"]],
            orientation="h",
            base=[(row["FIRST_STOCKOUT_AT"] - pd.Timestamp("2020-01-01")).days],
            marker_color=COLOR_RED,
            opacity=0.75,
            showlegend=False,
            hovertemplate=(
                f"<b>{row['CANONICAL_NAME']}</b><br>"
                f"From: {row['FIRST_STOCKOUT_AT'].date()}<br>"
                f"To: {row['LAST_STOCKOUT_AT'].date()}<br>"
                f"Duration: {duration}d<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT,
        height=max(300, top_n * 26),
        barmode="overlay",
        xaxis=dict(title="Days", gridcolor=_GRIDLINE["color"]),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ── Lead time histogram ───────────────────────────────────────────────────────

def anomaly_trend_chart(
    daily: pd.DataFrame,
    recent_start: "pd.Timestamp",
    baseline_avg: float,
    recent_avg: float,
    spike_start: "Optional[pd.Timestamp]" = None,
) -> go.Figure:
    """
    Mini consumption trend for an anomaly panel.
    daily: DataFrame(date, qty) covering baseline + recent window.
    Shades baseline period grey, recent period amber.
    Draws reference lines for baseline avg and current rate.
    """
    split = pd.to_datetime(recent_start)

    baseline_df = daily[daily["date"] <= split]
    recent_df   = daily[daily["date"] >  split]

    fig = go.Figure()

    # ── Baseline area ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=baseline_df["date"], y=baseline_df["qty"],
        fill="tozeroy", mode="lines",
        line=dict(color="#9CA3AF", width=1),
        fillcolor="rgba(156,163,175,0.15)",
        name="Baseline period",
        hovertemplate="%{x|%d %b}: %{y:.0f} units<extra>Baseline period</extra>",
    ))

    # ── Recent area ──────────────────────────────────────────────────────────
    # Bridge: include last baseline point so there's no gap at the split
    bridge = pd.concat([baseline_df.tail(1), recent_df])
    fig.add_trace(go.Scatter(
        x=bridge["date"], y=bridge["qty"],
        fill="tozeroy", mode="lines",
        line=dict(color="#D97706", width=1.5),
        fillcolor="rgba(217,119,6,0.12)",
        name="Recent 14 days",
        hovertemplate="%{x|%d %b}: %{y:.0f} units<extra>Recent</extra>",
    ))

    x_min = daily["date"].min()
    x_max = daily["date"].max()

    # ── Baseline avg reference line ──────────────────────────────────────────
    if baseline_avg > 0:
        fig.add_shape(type="line",
            x0=x_min, x1=split, y0=baseline_avg, y1=baseline_avg,
            line=dict(color="#9CA3AF", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x_min, y=baseline_avg,
            text=f" {baseline_avg:.1f}/d (baseline)",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="#9CA3AF"),
        )

    # ── Current rate reference line ──────────────────────────────────────────
    if recent_avg > 0:
        fig.add_shape(type="line",
            x0=split, x1=x_max, y0=recent_avg, y1=recent_avg,
            line=dict(color="#D97706", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x_max, y=recent_avg,
            text=f"{recent_avg:.1f}/d now ",
            showarrow=False, xanchor="right",
            font=dict(size=9, color="#D97706"),
        )

    # ── Spike start marker ───────────────────────────────────────────────────
    if spike_start is not None:
        fig.add_vline(
            x=spike_start.timestamp() * 1000,
            line=dict(color="#DC2626", width=1, dash="dash"),
            annotation_text="spike",
            annotation_font_size=9,
            annotation_font_color="#DC2626",
        )

    # ── Split line ───────────────────────────────────────────────────────────
    fig.add_vline(
        x=split.timestamp() * 1000,
        line=dict(color="#E5E7EB", width=1),
    )

    _layout = {**_LAYOUT, "margin": dict(l=0, r=0, t=4, b=0)}
    fig.update_layout(
        **_layout,
        height=150,
        showlegend=False,
        xaxis=dict(
            showgrid=False, showticklabels=True,
            tickformat="%d %b", tickfont=dict(size=9), nticks=6,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            showticklabels=True, tickfont=dict(size=9),
            rangemode="tozero",
        ),
    )
    return fig


def lead_time_histogram(lead_times: list[float], facility_name: str = "") -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=lead_times,
        nbinsx=20,
        marker_color=COLOR_PRIMARY,
        opacity=0.8,
        hovertemplate="Lead time: %{x:.0f}d<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        height=260,
        title=f"Lead time distribution — {facility_name}" if facility_name else "Lead time distribution",
        xaxis=dict(title="Days", gridcolor=_GRIDLINE["color"]),
        yaxis=dict(title="Observations", gridcolor=_GRIDLINE["color"]),
    )
    return fig
