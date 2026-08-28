"""Plotly figure builders for the SPH dashboard.

Every builder shows the numbers as they are — ranges and whiskers stay
visible so the reader sees how sure the estimate is. Color encodes magnitude
with one blue ramp, identity with the fixed categorical order, and state with
the reserved status colors (always paired with a text label). One y-axis per
chart, always.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from inventory_intelligence.dashboard import theme

_MASK_FILL = "rgba(137,135,129,0.18)"       # neutral wash for gaps in records
_CENSOR_FILL = "rgba(236,131,90,0.18)"      # amber wash for days stock was unavailable

_DOC_LABELS = {"ORDER": "ordered", "RECEIPT": "received"}

# ── Capital-state palette (working / excess / slow / dead) ─────────────────────
# Ordered severity, four distinct hues — validated CVD-safe in light and dark by
# the dataviz palette validator, and always shown with a direct segment label so
# identity never rests on colour alone.
_CAPITAL_LIGHT = {"working": "#1BAF7A", "excess": "#2A78D6",
                  "slow": "#EDA100", "dead": "#E34948"}
_CAPITAL_DARK = {"working": "#199E70", "excess": "#3987E5",
                 "slow": "#C98500", "dead": "#E34948"}
_CAPITAL_LABELS = {"working": "Working", "excess": "Surplus (above need)",
                   "slow": "Slow-moving", "dead": "Dead"}
_CAPITAL_ORDER = ["working", "excess", "slow", "dead"]


def _capital_colors() -> dict:
    return _CAPITAL_DARK if theme.is_dark() else _CAPITAL_LIGHT


# ── Surplus root-cause palette (why a surplus exists) ──────────────────────────
# Reuses the validated capital hues so the "why" reads in the same visual
# language: falling demand = the slow/cooling gold, a lot-size artifact = the
# surplus blue, steady overstock = the working green. No new colour to validate.
_CAUSE_ORDER = ["demand_fell", "over_bought", "steady_overstock"]
_CAUSE_LABELS = {"demand_fell": "Demand is falling",
                 "over_bought": "Bought too much at once",
                 "steady_overstock": "Above plan at steady use"}


def cause_palette() -> dict:
    """Mode-aware colour per surplus root cause (see ``_CAUSE_ORDER``)."""
    c = _capital_colors()
    return {"demand_fell": c["slow"], "over_bought": c["excess"],
            "steady_overstock": c["working"]}


def capital_states(health: pd.DataFrame) -> dict:
    """Partition priced inventory value (KES) into four non-overlapping states:

    - ``working`` — active stock up to its top-up level (capital genuinely at work)
    - ``excess``  — the over-top-up surplus of active items (releasable now)
    - ``slow``    — value in slow-moving items
    - ``dead``    — value in dead items

    Items whose stock count looks wrong (>2 years of supply — a likely receiving
    error) are NOT surfaced separately: their value stays inside ``working`` and
    their inflated surplus is excluded from ``excess``, so the releasable figure
    is never inflated by a phantom count. They sum to total priced inventory
    value — no bucket is double-counted.
    """
    h = health.copy()
    iv = pd.to_numeric(h.get("inventory_value"), errors="coerce").fillna(0.0)
    ex = pd.to_numeric(h.get("excess_value"), errors="coerce").fillna(0.0).clip(lower=0)
    mc = h.get("movement_class", pd.Series("", index=h.index)).astype(str)
    if "stock_suspect" in h.columns:
        susp = h["stock_suspect"].fillna(False).astype(bool)
    else:
        susp = pd.Series(False, index=h.index)
    ok = ~susp
    active = mc == "active"
    dead = float(iv[(mc == "dead") & ok].sum())
    slow = float(iv[(mc == "slow") & ok].sum())
    excess = float(ex[active & ok].sum())            # verified surplus only
    working = float(iv.sum() - excess - slow - dead)  # everything else, incl. suspect
    return {"working": max(working, 0.0), "excess": excess,
            "slow": slow, "dead": dead}


def _short(name: str, limit: int = 38) -> str:
    text = str(name)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _doc_label(value: str) -> str:
    return _DOC_LABELS.get(str(value).upper(), str(value))


# ── Chance of running out ─────────────────────────────────────────────────────

# ── Demand ────────────────────────────────────────────────────────────────────

def demand_history(
    daily: pd.DataFrame,
    title: str = "Daily quantity used",
) -> go.Figure:
    """One item's daily history, with days that stock was unavailable shaded."""
    df = daily.sort_values("date").copy()
    primary = theme.series_primary()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["quantity"], mode="lines",
            line=dict(width=2.5, color=primary), name="used / day",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} units<extra></extra>",
        )
    )

    def _windows(flags: pd.Series) -> list[tuple]:
        """Consecutive-day runs where a boolean flag holds."""
        active = df.loc[flags.fillna(False).astype(bool), "date"]
        if active.empty:
            return []
        runs, start, prev = [], active.iloc[0], active.iloc[0]
        for ts in active.iloc[1:]:
            if (ts - prev).days > 1:
                runs.append((start, prev))
                start = ts
            prev = ts
        runs.append((start, prev))
        return runs

    if "masked" in df.columns:
        for i, (lo, hi) in enumerate(_windows(df["masked"])):
            fig.add_vrect(
                x0=lo, x1=hi + pd.Timedelta(days=1), fillcolor=_MASK_FILL, line_width=0,
                annotation_text="no records" if i == 0 else None,
                annotation_position="top left",
                annotation_font=dict(size=10),
            )
    if "censored" in df.columns:
        for i, (lo, hi) in enumerate(_windows(df["censored"])):
            fig.add_vrect(
                x0=lo, x1=hi + pd.Timedelta(days=1), fillcolor=_CENSOR_FILL, line_width=0,
                annotation_text="stock was unavailable" if i == 0 else None,
                annotation_position="top right",
                annotation_font=dict(size=10),
            )
    fig.update_layout(
        title=title, yaxis_title="units / day", hovermode="x unified", height=340,
        showlegend=False,
    )
    return theme.apply(fig)


def forecast_fan(fc: pd.DataFrame, title: str = "Forecast — total expected use") -> go.Figure:
    """Forecast of total use over the horizon, with a widening confidence range.

    ``fc``: rows for one item with columns horizon, q05, q25, q50, q75, q95.
    The outer band is the confidence range, the inner band the likely range,
    and the line the expected total.
    """
    rows = fc.dropna(subset=["q50"]).sort_values("horizon").copy()
    if rows.empty:
        return theme.apply(go.Figure())
    x = [0] + rows["horizon"].tolist()
    primary = theme.series_primary()
    fill_outer = theme.rgba(primary, 0.13)   # confidence-range wash
    fill_inner = theme.rgba(primary, 0.26)   # likely-range wash

    def col(name: str) -> list:
        return [0.0] + rows[name].tolist()

    fig = go.Figure()
    # confidence range (outer)
    fig.add_trace(go.Scatter(x=x, y=col("q95"), mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=col("q05"), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=fill_outer,
                             name="confidence range", hoverinfo="skip"))
    # likely range (inner)
    fig.add_trace(go.Scatter(x=x, y=col("q75"), mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=col("q25"), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=fill_inner,
                             name="likely range", hoverinfo="skip"))
    # expected
    fig.add_trace(
        go.Scatter(
            x=x, y=col("q50"), mode="lines+markers",
            line=dict(width=2.5, color=primary), marker=dict(size=8),
            name="expected",
            hovertemplate="over %{x} days<br>expected %{y:.0f} units<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="days ahead",
        yaxis_title="total units", height=340, hovermode="x unified",
    )
    return theme.apply(fig)


# ── Grouping ──────────────────────────────────────────────────────────────────

# ── Restocking ────────────────────────────────────────────────────────────────

# ── Sizes ─────────────────────────────────────────────────────────────────────

# ── Forecast accuracy ─────────────────────────────────────────────────────────

def forecast_vs_actual(df: pd.DataFrame, name_col: str = "display_name") -> go.Figure:
    """Predicted vs actually-used units per item. Points on the diagonal were
    spot-on; the predicted range shows as a horizontal whisker; color marks
    whether actual use landed inside the predicted range."""
    data = df.dropna(subset=["forecast_expected", "actual"]).copy()
    if data.empty:
        return theme.apply(go.Figure())
    names = data[name_col].map(_short) if name_col in data.columns \
        else data.get("item_key", pd.Series(range(len(data)))).astype(str)
    data = data.assign(_name=list(names))

    lim = float(max(data["forecast_expected"].max(), data["actual"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(width=1, dash="dash", color=theme.chrome()["muted"]),
            name="spot-on", hoverinfo="skip",
        )
    )
    have_range = {"forecast_low", "forecast_high"} <= set(data.columns)
    for inside, color, label in [
        (True, theme.STATUS["good"], "within predicted range"),
        (False, theme.STATUS["serious"], "outside predicted range"),
    ]:
        sub = data[data["within_range"].astype(bool) == inside]
        if sub.empty:
            continue
        error_x = None
        if have_range:
            error_x = dict(
                type="data", symmetric=False,
                array=sub["forecast_high"] - sub["forecast_expected"],
                arrayminus=sub["forecast_expected"] - sub["forecast_low"],
                color=theme.chrome()["baseline"], thickness=1, width=0,
            )
        fig.add_trace(
            go.Scatter(
                x=sub["forecast_expected"], y=sub["actual"], mode="markers",
                name=label,
                marker=dict(size=9, color=color,
                            line=dict(width=2, color="rgba(0,0,0,0)")),
                error_x=error_x,
                text=sub["_name"],
                hovertemplate=(
                    "<b>%{text}</b><br>predicted %{x:.0f} units"
                    "<br>actually used %{y:.0f} units"
                    f"<br>{label}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Forecast vs actual use",
        xaxis_title="predicted (expected) units",
        yaxis_title="actually used units",
        height=380,
    )
    return theme.apply(fig)


# ── Spending & suppliers ──────────────────────────────────────────────────────

def spend_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly value, ordered vs received."""
    monthly = (
        df.groupby(["doc_month", "doc_type"], as_index=False)["total_value"].sum()
    )
    palette = theme.categorical()
    fig = go.Figure()
    for i, doc_type in enumerate(sorted(monthly["doc_type"].astype(str).unique())[:4]):
        sub = monthly[monthly["doc_type"] == doc_type].sort_values("doc_month")
        fig.add_trace(
            go.Scatter(
                x=sub["doc_month"], y=sub["total_value"], mode="lines+markers",
                name=_doc_label(doc_type), line=dict(width=2, color=palette[i]),
                marker=dict(size=7),
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f} KES<extra></extra>",
            )
        )
    fig.update_layout(
        title="Monthly spending — ordered vs received",
        yaxis_title="KES", hovermode="x unified", height=340,
    )
    return theme.apply(fig)


def supplier_pareto(df: pd.DataFrame, value_col: str = "received_value", top_n: int = 15) -> go.Figure:
    """Which suppliers account for most of the spend: share bars + running total."""
    data = df.groupby("supplier_name", as_index=False)[value_col].sum()
    data = data.sort_values(value_col, ascending=False)
    total = data[value_col].sum()
    if total <= 0:
        return theme.apply(go.Figure())
    data["share"] = data[value_col] / total
    data["cumshare"] = data["share"].cumsum()
    top = data.head(top_n)
    names = top["supplier_name"].fillna("(unknown)").map(_short)
    primary = theme.series_primary()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names, y=top["share"], name="share of value",
            marker=dict(color=primary, cornerradius=4,
                        line=dict(width=2, color="rgba(0,0,0,0)")),
            customdata=top[value_col],
            hovertemplate="<b>%{x}</b><br>%{y:.1%} · %{customdata:,.0f} KES<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=names, y=top["cumshare"], name="running total",
            mode="lines+markers", line=dict(width=2.5, color=theme.categorical()[1]),
            marker=dict(size=7),
            hovertemplate="running total %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top suppliers by {_doc_label(value_col.split('_')[0])} value "
              f"(top {len(top)} of {len(data)})",
        yaxis_title="share of total value", yaxis_tickformat=".0%",
        height=380,
    )
    return theme.apply(fig)


# ── Capital: where the money sits, and what's releasable ──────────────────────

def capital_decomposition_bar(health: pd.DataFrame) -> go.Figure:
    """One horizontal stacked bar splitting inventory value across the four
    capital states — the 'short and overstocked at once' picture in a glance.
    Each segment carries its own label, so state never rests on colour alone."""
    states = capital_states(health)
    total = sum(states.values())
    if total <= 0:
        return theme.apply(go.Figure())
    col = _capital_colors()
    surface = theme.chrome()["surface"]
    fig = go.Figure()
    for key in _CAPITAL_ORDER:
        val = states[key]
        if val <= 0:
            continue
        share = val / total
        fig.add_trace(go.Bar(
            x=[val], y=["capital"], orientation="h", name=_CAPITAL_LABELS[key],
            marker=dict(color=col[key], cornerradius=4,
                        line=dict(width=2, color=surface)),
            text=[f"{theme.fmt_kes_compact(val)} · {share * 100:.0f}%"]
                 if share >= 0.08 else [""],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff"),
            hovertemplate=(f"{_CAPITAL_LABELS[key]}<br>KES %{{x:,.0f}} · "
                           f"{share * 100:.1f}%<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack", height=170,
        margin=dict(l=10, r=24, t=18, b=52),
        xaxis=dict(visible=False),            # segment labels already carry KES + %
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    return theme.apply(fig)


def capital_waterfall(health: pd.DataFrame) -> go.Figure:
    """Total inventory value stepping down through the capital that can be
    released — each step in its own colour: value to verify, then dead, slow,
    and the over-top-up surplus, down to the working base. Built by hand (not
    go.Waterfall) so every step carries a distinct, meaningful colour."""
    s = capital_states(health)
    total = sum(s.values())
    if total <= 0:
        return theme.apply(go.Figure())
    col = _capital_colors()
    ch = theme.chrome()
    # running edges for the descending float bars
    r1 = total - s["dead"]; r2 = r1 - s["slow"]; r3 = r2 - s["excess"]   # r3 == working
    labels = ["On hand", "Dead", "Slow-moving", "Surplus", "Working base"]
    heights = [total, s["dead"], s["slow"], s["excess"], s["working"]]
    bases = [0, r1, r2, r3, 0]
    colors = [ch["ink_secondary"], col["dead"], col["slow"], col["excess"], col["working"]]
    keep = [i for i, h in enumerate(heights) if h > 0 or i in (0, len(heights) - 1)]
    fig = go.Figure(go.Bar(
        x=[labels[i] for i in keep], y=[heights[i] for i in keep],
        base=[bases[i] for i in keep],
        marker=dict(color=[colors[i] for i in keep], cornerradius=4,
                    line=dict(width=2, color=ch["surface"])),
        text=[theme.fmt_kes_compact(heights[i]) for i in keep],
        textposition="outside", textfont=dict(size=11), cliponaxis=False,
        hovertemplate="%{x}<br>KES %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="inventory value (KES)", height=360, showlegend=False,
        margin=dict(l=10, r=28, t=24, b=52),
    )
    return theme.apply(fig)


def surplus_capital_split_bars(release: pd.DataFrame, name_col: str = "display_name",
                               top_n: int = 15) -> go.Figure:
    """Per item, the capital tied up split into what's genuinely needed to stay
    in service (the working requirement) versus the releasable surplus held on
    top — with the surplus segment coloured by *why* it built up. Ranked by
    releasable KES, each bar annotated with how many months of supply it holds.
    The one picture that says 'you're holding N months of an item you use in M'."""
    d = release.copy()
    for c in ("excess_value", "working_requirement", "soh", "unit_price",
              "months_of_supply"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.get("excess_value", pd.Series(dtype=float)).fillna(0) > 0]
    d = d[d["unit_price"].notna() & (d["unit_price"] > 0)]   # priced: capital is splittable
    if d.empty:
        return theme.apply(go.Figure())
    d["needed_value"] = (d["working_requirement"].clip(lower=0) * d["unit_price"]).clip(lower=0)
    d = d.sort_values("excess_value", ascending=False).head(top_n).iloc[::-1]
    names = (d[name_col] if name_col in d.columns else d["item_key"]).map(_short)
    colc = cause_palette()
    ch = theme.chrome()

    fig = go.Figure()
    # In-service need — the muted base of the stack (not up for release).
    fig.add_trace(go.Bar(
        x=d["needed_value"], y=names, orientation="h", name="In-service need",
        marker=dict(color=ch["baseline"], cornerradius=4,
                    line=dict(width=2, color="rgba(0,0,0,0)")),
        hovertemplate="<b>%{y}</b><br>needed in service: KES %{x:,.0f}<extra></extra>",
    ))
    # Releasable surplus — one segment per cause, so colour carries the 'why'.
    has_cause = "cause" in d.columns
    causes = _CAUSE_ORDER if has_cause else ["_all"]
    for key in causes:
        if has_cause:
            mask = (d["cause"] == key).to_numpy()
        else:
            mask = np.ones(len(d), dtype=bool)
        if not mask.any():
            continue
        xs = [v if m else None for v, m in zip(d["excess_value"], mask)]
        mos = [f"{mo:.0f}m held" if (m and pd.notna(mo)) else ""
               for m, mo in zip(mask, d["months_of_supply"])]
        txt = [theme.fmt_kes_compact(v) if m else "" for v, m in zip(d["excess_value"], mask)]
        fig.add_trace(go.Bar(
            x=xs, y=names, orientation="h", cliponaxis=False,
            name=(_CAUSE_LABELS[key] if has_cause else "Releasable surplus"),
            marker=dict(color=(colc[key] if has_cause else _capital_colors()["excess"]),
                        cornerradius=4, line=dict(width=2, color="rgba(0,0,0,0)")),
            text=txt, textposition="outside", customdata=mos,
            hovertemplate=("<b>%{y}</b><br>releasable: KES %{x:,.0f}"
                           "<br>%{customdata}<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack", xaxis_title="capital tied up (KES) — needed vs releasable",
        height=max(300, 32 * len(d) + 80),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
        margin=dict(r=110, t=20, b=60),
    )
    return theme.apply(fig)


def anomaly_flagged_bars(df: pd.DataFrame, name_col: str = "display_name",
                         top_n: int = 20) -> go.Figure:
    """Only the items whose recent use is genuinely off-pattern, as a diverging
    bar: how far above (using much more) or below (using much less) their usual
    level each one is. Points the eye at the handful that need a decision, not
    the thousand that don't."""
    d = df.copy()
    if "fdr_flag" in d.columns:
        d = d[d["fdr_flag"].fillna(False).astype(bool)]
    rr = pd.to_numeric(d.get("rate_ratio_window"), errors="coerce")
    d = d.assign(pct=(rr - 1.0) * 100.0).dropna(subset=["pct"])
    if d.empty:
        return theme.apply(go.Figure())
    d = d.reindex(d["pct"].abs().sort_values().index).tail(top_n)
    names = (d[name_col] if name_col in d.columns else d["item_key"]).map(_short)
    spike, collapse = theme.status()["serious"], theme.categorical()[2]
    colors = [spike if v > 0 else collapse for v in d["pct"]]
    labels = [("+" if v > 0 else "") + f"{v:,.0f}%" for v in d["pct"]]
    fig = go.Figure(go.Bar(
        x=d["pct"], y=names, orientation="h", cliponaxis=False,
        marker=dict(color=colors, cornerradius=4,
                    line=dict(width=2, color="rgba(0,0,0,0)")),
        text=labels, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}% vs its usual use<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="recent use vs usual (%)  ·  left = using less, right = using more",
        height=max(240, 30 * len(d) + 70), showlegend=False,
        margin=dict(l=10, r=70, t=20, b=56),
    )
    fig.add_vline(x=0, line=dict(color=theme.chrome()["baseline"], width=1))
    return theme.apply(fig)


def nonmoving_value_bars(df: pd.DataFrame, name_col: str = "display_name",
                         top_n: int = 15) -> go.Figure:
    """Top slow/dead items by value tied up, colour-coded by movement class —
    where to focus a write-off or redeployment decision first."""
    d = df.copy()
    d["inventory_value"] = pd.to_numeric(d.get("inventory_value"), errors="coerce")
    d = d.dropna(subset=["inventory_value"]).sort_values(
        "inventory_value", ascending=False).head(top_n)
    if d.empty:
        return theme.apply(go.Figure())
    d = d.iloc[::-1]
    names = (d[name_col] if name_col in d.columns else d["item_key"]).map(_short)
    col = _capital_colors()
    fig = go.Figure()
    for key in ("dead", "slow"):
        xs = [v if mc == key else None
              for v, mc in zip(d["inventory_value"], d["movement_class"])]
        if all(v is None for v in xs):
            continue
        fig.add_trace(go.Bar(
            x=xs, y=names, orientation="h", name=_CAPITAL_LABELS[key], cliponaxis=False,
            marker=dict(color=col[key], cornerradius=4,
                        line=dict(width=2, color="rgba(0,0,0,0)")),
            text=[theme.fmt_kes_compact(v) if v is not None else "" for v in xs],
            textposition="outside",
            hovertemplate=(f"<b>%{{y}}</b><br>{_CAPITAL_LABELS[key]}: "
                           "KES %{x:,.0f}<extra></extra>"),
        ))
    fig.update_layout(
        xaxis_title="value tied up (KES)", barmode="overlay",
        height=max(300, 30 * len(d) + 70),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
        margin=dict(r=80, t=20, b=56),
    )
    return theme.apply(fig)


# ── Capital efficiency: velocity × value ──────────────────────────────────────

def capital_efficiency_matrix(health: pd.DataFrame,
                              name_col: str = "display_name") -> go.Figure:
    """Velocity × inventory value — the 'capital efficiency' map. Each item is
    placed by how fast it turns (x) against how much capital it holds (y, log).
    Data-driven median crosshairs split it into four quadrants: fast+valuable
    (protect availability), slow+valuable (trapped capital — act), and the two
    low-value corners (routine / minor). Colour marks movement class, so a
    valuable item drifting into 'slow' or 'dead' is visible at a glance."""
    d = health.copy()
    d["itr"] = pd.to_numeric(d.get("itr"), errors="coerce")
    d["inventory_value"] = pd.to_numeric(d.get("inventory_value"), errors="coerce")
    d = d[(d["inventory_value"] > 0) & d["itr"].notna()]
    if d.empty:
        return theme.apply(go.Figure())
    movers = d.loc[d["itr"] > 0, "itr"]
    vx = float(movers.median()) if not movers.empty else float(d["itr"].median())
    vy = float(d["inventory_value"].median())
    col = _capital_colors()
    ch = theme.chrome()
    class_color = {"active": col["working"], "slow": col["slow"], "dead": col["dead"]}
    class_label = {"active": "In active use", "slow": "Slow-moving", "dead": "Dead"}

    fig = go.Figure()
    for key in ("active", "slow", "dead"):
        sub = d[d["movement_class"].astype(str) == key]
        if sub.empty:
            continue
        names = (sub[name_col] if name_col in sub.columns else sub["item_key"]).map(_short)
        fig.add_trace(go.Scatter(
            x=sub["itr"].clip(lower=0.01), y=sub["inventory_value"], mode="markers",
            name=class_label[key], text=names,
            marker=dict(size=10, color=class_color[key], opacity=0.75,
                        line=dict(width=1, color=ch["surface"])),
            hovertemplate=("<b>%{text}</b><br>turns %{x:.1f}×/yr"
                           "<br>KES %{y:,.0f} held<extra></extra>"),
        ))
    fig.add_vline(x=max(vx, 0.01), line=dict(color=ch["baseline"], width=1, dash="dot"))
    fig.add_hline(y=vy, line=dict(color=ch["baseline"], width=1, dash="dot"))
    for x, y, txt, anchor in [
        (0.015, 0.98, "Trapped capital — valuable but slow", "left"),
        (0.985, 0.98, "Protect — fast &amp; valuable", "right"),
        (0.015, 0.04, "Minor idle", "left"),
        (0.985, 0.04, "Efficient routine", "right"),
    ]:
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=txt,
                           showarrow=False, xanchor=anchor,
                           font=dict(size=11, color=ch["ink_secondary"]),
                           bgcolor=theme.rgba(ch["surface"], 0.7))
    fig.update_layout(
        xaxis_title="turnover — times used per year  (velocity →, log scale)",
        yaxis_title="capital held (KES, log scale)",
        xaxis_type="log", yaxis_type="log", height=480,
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
        margin=dict(l=10, r=28, t=30, b=76),
    )
    return theme.apply(fig)


# ── Supplier dependency ───────────────────────────────────────────────────────

def observed_supplier_bars(repeat: pd.DataFrame) -> go.Figure:
    """Distinct observed suppliers among REPEAT-purchased items only (1 / 2 / 3+).
    Establishes the base for the exposure table; deliberately compact. The
    single-observed-supplier bar carries the amber warning colour — that is the
    dependency to examine, not (yet) a problem."""
    n = pd.to_numeric(repeat.get("observed_suppliers"), errors="coerce")
    vals = [int((n == 1).sum()), int((n == 2).sum()), int((n >= 3).sum())]
    cats = ["1 (single observed)", "2", "3 or more"]
    warn = theme.status()["warning"]
    prim = theme.series_primary()
    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker=dict(color=[warn, prim, prim], cornerradius=4,
                    line=dict(width=2, color="rgba(0,0,0,0)")),
        text=[f"{v:,}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{x} supplier(s)<br>%{y:,} items<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="observed suppliers per item  (repeat-purchased only)",
        yaxis_title="items", height=260, showlegend=False,
        xaxis=dict(type="category"),   # 3 labelled buckets, not a numeric axis
        margin=dict(l=10, r=20, t=24, b=54),
    )
    return theme.apply(fig)
