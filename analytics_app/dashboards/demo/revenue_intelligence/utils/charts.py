"""
Plotly chart builders for the Revenue Intelligence dashboard.
House style: transparent paper/plot background, Afya design-system palette,
minimal chrome. Every builder returns a go.Figure to render with
st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils.formatting import fmt_kes_millions

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL   = "#0F6E56"
BLUE   = "#0C447C"
AMBER  = "#B45309"
RED    = "#B42318"
CRIT   = "#B42318"   # undispatched / our delay
SERIOUS = "#B45309"  # dispatched-unpaid / insurer delay
GREY   = "#9CA3AF"
INK    = "#1A1A2E"
GRID   = "#E5E7EB"

_LAYOUT = dict(
    font_family="sans-serif",
    font_color=INK,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=28, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
    hoverlabel=dict(font_size=12),
)


def _mlabels(months) -> list[str]:
    return [pd.to_datetime(m).strftime("%b '%y") for m in months]


# ── Revenue trend (invoiced + rolling avg, in-flight month flagged) ───────────

def revenue_trend(dm: pd.DataFrame, value_col="TOTAL_INVOICED", roll_col="ROLL_3M",
                  height=300) -> go.Figure:
    x = _mlabels(dm["REV_MONTH"])
    y = dm[value_col].tolist()
    complete = dm["IS_COMPLETE"].tolist()

    fig = go.Figure()
    # area fill under the solid (complete) portion
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=TEAL, width=2.6, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor="rgba(15,110,86,0.10)",
        name="Invoiced", hovertemplate="%{x}<br><b>%{customdata}</b><extra></extra>",
        customdata=[fmt_kes_millions(v) for v in y],
    ))
    # rolling average
    if roll_col in dm.columns:
        fig.add_trace(go.Scatter(
            x=x, y=dm[roll_col].tolist(), mode="lines",
            line=dict(color=INK, width=1.4, dash="dot"),
            name="3-mo average", hoverinfo="skip",
        ))
    # in-flight marker on the last point if incomplete
    if not complete[-1]:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers",
            marker=dict(color="#FFFFFF", size=11, line=dict(color=SERIOUS, width=2.5)),
            name="In-flight (partial)",
            hovertemplate=f"{x[-1]}<br><b>{fmt_kes_millions(y[-1])}</b><br>Partial month — still in-flight<extra></extra>",
        ))
    fig.update_layout(**_LAYOUT, height=height, showlegend=True,
                      xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                      yaxis=dict(gridcolor=GRID, zeroline=False, tickprefix="KES ", ticksuffix="",
                                 tickformat=".2s", tickfont=dict(size=10)))
    return fig


# ── Volume vs intensity decomposition ─────────────────────────────────────────

def driver_bars(dm: pd.DataFrame, n=8, height=260) -> go.Figure:
    d = dm.dropna(subset=["VOLUME_EFFECT"]).tail(n)
    x = _mlabels(d["REV_MONTH"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=d["VOLUME_EFFECT"] / 1e6, name="Volume effect",
                         marker_color=BLUE,
                         hovertemplate="%{x}<br>Volume: KES %{y:.2f}M<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=d["INTENSITY_EFFECT"] / 1e6, name="Intensity (case value)",
                         marker_color=TEAL,
                         hovertemplate="%{x}<br>Intensity: KES %{y:.2f}M<extra></extra>"))
    fig.update_layout(**_LAYOUT, height=height, barmode="relative", showlegend=True,
                      xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                      yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor="#C6CCC7",
                                 ticksuffix="M", tickprefix="KES ", tickfont=dict(size=10)))
    return fig


# ── Exposure split (two-clock, single stacked bar) ────────────────────────────

def exposure_split(undispatched: float, dispatched: float, height=120) -> go.Figure:
    total = undispatched + dispatched or 1
    fig = go.Figure()
    for name, val, col in [
        (f"Undispatched · our delay ({undispatched/total*100:.0f}%)", undispatched, CRIT),
        (f"Dispatched, unpaid · insurer delay ({dispatched/total*100:.0f}%)", dispatched, SERIOUS),
    ]:
        fig.add_trace(go.Bar(
            y=["AR"], x=[val], orientation="h", name=name, marker_color=col,
            text=fmt_kes_millions(val), textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#fff", size=13),
            hovertemplate=f"<b>{name}</b><br>{fmt_kes_millions(val)}<extra></extra>",
        ))
    _lay = {k: v for k, v in _LAYOUT.items() if k != "legend"}
    fig.update_layout(**_lay, height=height, barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=-0.35, x=0, font_size=11),
                      xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                      yaxis=dict(showticklabels=False, showgrid=False))
    return fig


# ── Two-clock accountability per insurer ──────────────────────────────────────

def two_clock(ar: pd.DataFrame, top=8, height=320) -> go.Figure:
    a = ar.copy()
    a["OUTSTANDING_KES"] = pd.to_numeric(a["OUTSTANDING_KES"], errors="coerce").fillna(0)
    piv = a.pivot_table(index="INSURER_LABEL", columns="AR_STATE",
                        values="OUTSTANDING_KES", aggfunc="sum", fill_value=0)
    und_col = next((c for c in piv.columns if "Undispatched" in c), None)
    dsp_col = next((c for c in piv.columns if "Dispatched" in c), None)
    piv["_tot"] = piv.sum(axis=1)
    piv = piv.sort_values("_tot", ascending=True).tail(top)
    labels = [c.replace("Social Health Authority (SHA)", "SHA").replace("⚠ Unknown (NULL company_id)", "Unknown insurer")
              .replace(" Insurance", "").replace(" General", "") for c in piv.index]
    fig = go.Figure()
    if und_col is not None:
        fig.add_trace(go.Bar(y=labels, x=piv[und_col], orientation="h", name="Undispatched · our delay",
                             marker_color=CRIT,
                             hovertemplate="%{y}<br>Undispatched: KES %{x:,.0f}<extra></extra>"))
    if dsp_col is not None:
        fig.add_trace(go.Bar(y=labels, x=piv[dsp_col], orientation="h", name="Dispatched, unpaid · insurer delay",
                             marker_color=SERIOUS,
                             hovertemplate="%{y}<br>Dispatched: KES %{x:,.0f}<extra></extra>"))
    fig.update_layout(**_LAYOUT, height=height, barmode="stack", showlegend=True,
                      xaxis=dict(gridcolor=GRID, tickprefix="KES ", tickformat=".2s", tickfont=dict(size=10)),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


# ── Dispatch-rate trend with collapse shading ─────────────────────────────────

def dispatch_trend(df: pd.DataFrame, go_live="2024-09-01", height=280) -> go.Figure:
    d = df.copy()
    d["REV_MONTH"] = pd.to_datetime(d["REV_MONTH"])
    d = d[d["REV_MONTH"] >= pd.Timestamp(go_live)].sort_values("REV_MONTH")
    x = _mlabels(d["REV_MONTH"])
    y = pd.to_numeric(d["DISPATCH_RATE_PCT"], errors="coerce").tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                             line=dict(color=BLUE, width=2.4), marker=dict(size=5),
                             name="Dispatch rate",
                             hovertemplate="%{x}<br>Dispatch rate: %{y:.0f}%<extra></extra>"))
    # shade the collapse region (trailing run of zeros)
    zero_run = 0
    for v in y[::-1]:
        if v == 0:
            zero_run += 1
        else:
            break
    if zero_run:
        fig.add_vrect(x0=x[len(x) - zero_run], x1=x[-1], fillcolor=RED, opacity=0.06,
                      line_width=0, annotation_text="0% dispatched", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=RED)
    fig.update_layout(**_LAYOUT, height=height, showlegend=False,
                      xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                      yaxis=dict(gridcolor=GRID, range=[0, 100], ticksuffix="%", tickfont=dict(size=10)))
    return fig


# ── Pareto (concentration) ────────────────────────────────────────────────────

def pareto(labels, values, height=300, bar_color=TEAL) -> go.Figure:
    s = pd.DataFrame({"l": labels, "v": values}).sort_values("v", ascending=False)
    cum = s["v"].cumsum() / s["v"].sum() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["l"], y=s["v"], marker_color=bar_color, name="Open AR",
                         hovertemplate="%{x}<br>KES %{y:,.0f}<extra></extra>", yaxis="y"))
    fig.add_trace(go.Scatter(x=s["l"], y=cum, mode="lines+markers", line=dict(color=AMBER, width=2),
                             marker=dict(size=5), name="Cumulative %", yaxis="y2",
                             hovertemplate="%{x}<br>Cumulative %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=80, line_dash="dash", line_color=GREY, yref="y2")
    fig.update_layout(**_LAYOUT, height=height, showlegend=True,
                      xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=9)),
                      yaxis=dict(gridcolor=GRID, tickprefix="KES ", tickformat=".2s", tickfont=dict(size=10)),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%",
                                  showgrid=False, tickfont=dict(size=10)))
    return fig


# ── Donut ─────────────────────────────────────────────────────────────────────

def donut(names, values, colors, center_big="", center_small="", height=230) -> go.Figure:
    fig = go.Figure(go.Pie(labels=names, values=values, hole=0.62, sort=False,
                           marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                           textinfo="none",
                           hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"))
    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "margin"},
                      margin=dict(l=0, r=0, t=6, b=6), height=height, showlegend=False,
                      annotations=[dict(text=f"<b>{center_big}</b>", x=0.5, y=0.54, font_size=20, showarrow=False),
                                   dict(text=center_small, x=0.5, y=0.40, font_size=11, font_color=GREY, showarrow=False)])
    return fig


# ── Channel mix (100% stacked over time) ──────────────────────────────────────

_CHANNEL_COLORS = {"M-Pesa": TEAL, "PesaPal": "#1D9E75", "Card": BLUE,
                   "Patient Account": "#6B7280", "Cheque": "#C084FC", "Cash": AMBER}


def channel_mix(coll: pd.DataFrame, height=280) -> go.Figure:
    d = coll.copy()
    d["REV_MONTH"] = pd.to_datetime(d["REV_MONTH"])
    d = d[d["TOTAL_COLLECTED"] > 0].sort_values("REV_MONTH")
    x = _mlabels(d["REV_MONTH"])
    chans = [("M-Pesa", "MPESA"), ("PesaPal", "PESAPAL"), ("Card", "CARD"),
             ("Patient Account", "PATIENT_ACCOUNT"), ("Cheque", "CHEQUE"), ("Cash", "CASH")]
    tot = d[[c for _, c in chans]].apply(pd.to_numeric, errors="coerce").sum(axis=1).replace(0, 1)
    fig = go.Figure()
    for name, col in chans:
        share = pd.to_numeric(d[col], errors="coerce").fillna(0) / tot * 100
        fig.add_trace(go.Bar(x=x, y=share, name=name, marker_color=_CHANNEL_COLORS[name],
                             hovertemplate=f"%{{x}}<br>{name}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(**_LAYOUT, height=height, barmode="stack",
                      xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                      yaxis=dict(gridcolor=GRID, ticksuffix="%", range=[0, 100], tickfont=dict(size=10)))
    return fig


# ── Service-line share shift (diverging) ──────────────────────────────────────

def svc_shift(names, shifts, height=240) -> go.Figure:
    s = pd.DataFrame({"n": names, "v": shifts}).sort_values("v")
    colors = [TEAL if v >= 0 else RED for v in s["v"]]
    fig = go.Figure(go.Bar(y=s["n"], x=s["v"], orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}pp" for v in s["v"]], textposition="outside",
                           textfont=dict(size=11),
                           hovertemplate="%{y}<br>Share shift: %{x:+.1f}pp<extra></extra>"))
    fig.update_layout(**_LAYOUT, height=height, showlegend=False,
                      xaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor="#C6CCC7",
                                 ticksuffix="pp", tickfont=dict(size=10)),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


# ── Simple horizontal bar ─────────────────────────────────────────────────────

def hbar(names, values, color=TEAL, height=260, value_fmt="kes") -> go.Figure:
    s = pd.DataFrame({"n": names, "v": values}).sort_values("v")
    if value_fmt == "kes":
        txt = [fmt_kes_millions(v) for v in s["v"]]
        htmpl = "%{y}<br>KES %{x:,.0f}<extra></extra>"
    else:
        txt = [f"{v:.0f}" for v in s["v"]]
        htmpl = "%{y}<br>%{x}<extra></extra>"
    fig = go.Figure(go.Bar(y=s["n"], x=s["v"], orientation="h", marker_color=color,
                           text=txt, textposition="auto", textfont=dict(size=11), hovertemplate=htmpl))
    fig.update_layout(**_LAYOUT, height=height, showlegend=False,
                      xaxis=dict(gridcolor=GRID, tickfont=dict(size=10), showticklabels=False),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


# ── AR aging (severity ramp: fresh → aged) ────────────────────────────────────

def aging_bar(dist: pd.DataFrame, height=230) -> go.Figure:
    """Undispatched AR by age bucket. dist has AGING_BUCKET, KES, SHARE (ordered)."""
    ramp = {"0–30 days": "#1D9E75", "31–60 days": "#EDA100",
            "61–90 days": "#B45309", "90+ days": "#B42318"}
    colors = [ramp.get(b, TEAL) for b in dist["AGING_BUCKET"]]
    fig = go.Figure(go.Bar(
        x=dist["AGING_BUCKET"], y=dist["KES"], marker_color=colors,
        text=[f"{fmt_kes_millions(v)}<br>{s*100:.0f}%" for v, s in zip(dist["KES"], dist["SHARE"])],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>KES %{y:,.0f}<extra></extra>"))
    fig.update_layout(**_LAYOUT, height=height, showlegend=False,
                      xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                      yaxis=dict(gridcolor=GRID, tickprefix="KES ", tickformat=".2s",
                                 tickfont=dict(size=10), rangemode="tozero"))
    return fig


# ── Theatre funnel ────────────────────────────────────────────────────────────

def theatre_funnel(booked, scheduled, completed, billed, height=240) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=["Booked", "Scheduled", "Completed", "Billed"],
        x=[booked, scheduled, completed, billed],
        marker=dict(color=[BLUE, "#1D9E75", TEAL, "#0A4B3A"]),
        textinfo="value+percent initial", textfont=dict(size=12),
        connector=dict(line=dict(color=GRID, width=1)),
    ))
    fig.update_layout(**_LAYOUT, height=height, showlegend=False)
    return fig
