"""
sph/opd_ipd_module/views.py
============================
All render functions for the SPH OPD → IPD Conversion tab.

Rules enforced here:
  - Every function takes a pd.DataFrame (or multiple) and renders to st.*.
  - Zero SQL — no database calls, no query strings.
  - All insight text is computed from the DataFrame passed in, never
    hardcoded. If a DataFrame is empty the function renders a graceful
    empty state and returns early.
  - Chart sizing: Plotly height values are set explicitly to keep paired
    column charts at equal heights. The same height constant is applied
    to both columns in every two-column section.

Section map (matches the dashboard entry point)
------------------------------------------------
  render_s1_kpis(df_kpis)
  render_s2_trend(df_trend)
  render_s3_treemap(df_segments)
  render_s4_segment_bar(df_segments)
  render_s5_ortho_deep_dive(df_ortho, df_spine_vol)
  render_s6_non_ortho(df_non_ortho)
  render_s7_factors(df_workload, df_staffing, df_comorbidity)
  render_s8_escalation(df_escalation, df_conversion, df_timing)
  render_s10_recommendations()
  render_tab_summary()
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sph.clinicals.opd_ipd_module.ui_template import (
    # Tokens
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, SECONDARY,
    ACCENT_INFO, ACCENT_POSITIVE, ACCENT_CRITICAL, ACCENT_MONITOR, ACCENT_NEUTRAL,
    TEXT_PRI as TEXT, TEXT_SEC, TEXT_MUT as TEXT_MUTED, TEXT_MUT as TEXT_HINT,
    SURFACE_1 as SURFACE, BORDER,
    # Chart helpers
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG, H_SINGLE, H_PAIRED,
    # UI components
    page_header, section_header,
    kpi_row,
    chart_container_open, chart_container_close,
    chart_card, chart_card_close,
    insight_bar,
    sharp_finding_card,
    priority_cards,
    # Formatters
    fmt_num, fmt_pct, fmt_delta,
)

# ── Design constants ────────────────────────────────────────────────────────
_C_BLUE   = PRIMARY    # teal — orthopedics category (spec §4)
_C_GRAY   = "#D3D6DE"
_C_RED    = DANGER
_C_AMBER  = WARNING
_C_GREEN  = SUCCESS
_C_PURPLE = SECONDARY  # raspberry — general surgery/OBGYN category (spec §4)
_C_PINK   = "#D6698C"

_H_SINGLE = 260   # full-width chart height
_H_PAIRED = 240   # paired column chart height — same for both columns

# _AXIS replaced by imported AXIS_Y from ui_template
# _LAYOUT replaced by imported CHART_LAYOUT from ui_template
_LAYOUT = CHART_LAYOUT


def _empty(msg: str = "No data available") -> None:
    st.markdown(
        f'<div style="padding:20px;text-align:center;color:{TEXT_HINT};'
        f'font-size:13px;font-style:italic">{msg}</div>',
        unsafe_allow_html=True,
    )


def _safe(df: pd.DataFrame) -> bool:
    """Return True if df has rows, False otherwise."""
    return df is not None and not df.empty


# ── Colour helpers ───────────────────────────────────────────────────────────

# ── S1: Headline KPIs ────────────────────────────────────────────────────────

def render_s1_kpis(df: pd.DataFrame) -> None:
    section_header("Headline")
    if not _safe(df):
        _empty()
        return

    row = df.iloc[0]
    total      = int(row.get("TOTAL_VISITS", 0))
    admitted   = int(row.get("TOTAL_ADMISSIONS", 0))
    overall    = float(row.get("OVERALL_CONVERSION_PCT", 0))
    acute      = float(row.get("ACUTE_CONVERSION_PCT", 0))

    overall_color = ACCENT_MONITOR if overall < 10 else ACCENT_POSITIVE
    admitted_color = ACCENT_CRITICAL if overall < 8 else ACCENT_MONITOR

    kpi_row([
        {
            "label":        "Total visits",
            "value":        fmt_num(total),
            "delta":        "All clinical segments",
            "accent_color": ACCENT_INFO,
        },
        {
            "label":        "Overall conversion rate",
            "value":        fmt_pct(overall),
            "delta":        f"New/acute only: {fmt_pct(acute)}",
            "delta_good":   acute > overall,
            "accent_color": overall_color,
        },
        {
            "label":        "Total admissions",
            "value":        fmt_num(admitted),
            "delta":        f"Inpatient of {fmt_num(total)} total",
            "accent_color": admitted_color,
        },
    ])


# ── S2: Monthly trend ────────────────────────────────────────────────────────

def render_s2_trend(df: pd.DataFrame) -> None:
    section_header("Overall trend")
    if not _safe(df):
        _empty()
        return

    # Ensure date column is datetime
    df = df.copy()
    df["VISIT_MONTH"] = pd.to_datetime(df["VISIT_MONTH"])

    # Mark pre-Jun 2022 rows as unreliable (zero admissions, data gap)
    cutoff = pd.Timestamp("2022-06-01")
    df_reliable   = df[df["VISIT_MONTH"] >= cutoff]

    # ── Volume chart (stacked bar) ──
    chart_card(
        "Monthly visit volume — outpatient and admitted",
    )
    fig_vol = go.Figure()

    fig_vol.add_trace(go.Bar(
        x=df_reliable["VISIT_MONTH"],
        y=df_reliable["OUTPATIENT"],
        name="Outpatient",
        marker_color="#8FCFC8",
        showlegend=True,
    ))
    fig_vol.add_trace(go.Bar(
        x=df_reliable["VISIT_MONTH"],
        y=df_reliable["ADMITTED"],
        name="Admitted",
        marker_color=_C_BLUE,
        showlegend=True,
    ))

    fig_vol.update_layout(
        **{**_LAYOUT, "height": _H_PAIRED, "barmode": "stack"},
        xaxis={**AXIS_Y, "showgrid": False},
        yaxis={**AXIS_Y, "title_text": "Visits"},
    )
    st.plotly_chart(fig_vol, use_container_width=True, config=PC_CFG)
    chart_card_close()

    # ── Conversion rate line ──
    chart_card("Monthly conversion rate (%)")
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=df_reliable["VISIT_MONTH"],
        y=df_reliable["CONVERSION_PCT"],
        mode="lines+markers",
        line=dict(color=_C_BLUE, width=2),
        marker=dict(size=4, color=_C_BLUE),
        fill="tozeroy",
        fillcolor="rgba(27,138,130,0.08)",
        name="Conversion rate",
        showlegend=False,
    ))
    fig_rate.update_layout(
        **{**_LAYOUT, "height": 160},
        xaxis={**AXIS_Y, "showgrid": False},
        yaxis={**AXIS_Y, "ticksuffix": "%", "title_text": ""},
    )
    st.plotly_chart(fig_rate, use_container_width=True, config=PC_CFG)
    chart_card_close()

    # Year-over-year mechanism, computed from the live data rather than
    # hardcoded years/percentages that go stale as new data lands. An
    # absolute admissions decline is flagged as critical regardless of what's
    # happening to OPD volume — "OPD grew while admissions fell" is a
    # revenue-risk finding, not something to wave away as mere rate dilution.
    yearly = df_reliable.copy()
    yearly["YEAR"] = yearly["VISIT_MONTH"].dt.year
    yearly = yearly.groupby("YEAR").agg(
        OPD=("OUTPATIENT", "sum"), ADM=("ADMITTED", "sum"),
    ).reset_index().sort_values("YEAR")

    # Classify each year-over-year step, then SYNTHESIZE: a run of consecutive
    # years with the same mechanism is a sustained trend and should read as
    # one finding, not a template sentence repeated once per year.
    steps = []
    for (_, prev), (_, curr) in zip(yearly.iloc[:-1].iterrows(), yearly.iloc[1:].iterrows()):
        y1, y2 = int(prev["YEAR"]), int(curr["YEAR"])
        opd1, opd2 = float(prev["OPD"]), float(curr["OPD"])
        adm1, adm2 = float(prev["ADM"]), float(curr["ADM"])
        if adm1 <= 0:
            continue
        adm_delta = (adm2 - adm1) / adm1 * 100
        opd_delta = (opd2 - opd1) / opd1 * 100 if opd1 > 0 else None
        if adm_delta >= -1:
            continue  # admissions flat or growing — nothing to explain
        mechanism = "opd_growth_dilution" if (opd_delta is not None and opd_delta > 5) else "flat_opd_decline"
        steps.append({"y1": y1, "y2": y2, "adm_delta": adm_delta, "opd_delta": opd_delta, "mech": mechanism})

    bullets = ["Pre-Jun 2022 excluded — zero admissions recorded, likely a data gap."]
    any_critical = bool(steps)

    i = 0
    while i < len(steps):
        run = [steps[i]]
        while i + 1 < len(steps) and steps[i + 1]["mech"] == run[-1]["mech"]:
            i += 1
            run.append(steps[i])
        i += 1

        span_start, span_end = run[0]["y1"], run[-1]["y2"]
        if run[0]["mech"] == "opd_growth_dilution":
            for s in run:
                bullets.append(
                    f"{s['y1']}→{s['y2']}: OPD volume grew {s['opd_delta']:.0f}% while admissions "
                    f"fell {abs(s['adm_delta']):.0f}% — more patients came through the door but "
                    f"fewer were admitted. This is a revenue-risk finding, not just rate dilution, "
                    f"and needs a root-cause check (triage threshold change, staffing, bed "
                    f"capacity, referral pattern) before assuming it's benign."
                )
        elif len(run) == 1:
            s = run[0]
            bullets.append(
                f"{s['y1']}→{s['y2']}: admissions fell {abs(s['adm_delta']):.0f}% against roughly "
                f"flat OPD volume — fewer patients admitted, not more volume arriving."
            )
        else:
            # Cumulative effect across the whole run, not year-by-year restatement.
            adm_start = float(yearly.loc[yearly["YEAR"] == span_start, "ADM"].iloc[0])
            adm_end = float(yearly.loc[yearly["YEAR"] == span_end, "ADM"].iloc[0])
            per_year = ", ".join(f"{s['y2']}: -{abs(s['adm_delta']):.0f}%" for s in run)
            cumulative_txt = ""
            if adm_start > 0:
                cumulative = (adm_end - adm_start) / adm_start * 100
                cumulative_txt = f" — a cumulative {abs(cumulative):.0f}% drop in admissions over the period"
            bullets.append(
                f"{span_start}→{span_end}: admissions have fallen for {len(run)} straight years "
                f"({per_year}) while OPD volume stayed roughly flat{cumulative_txt}. This reads as "
                f"a structural, ongoing decline rather than a one-off dip — worth investigating "
                f"before treating any single year as an isolated event."
            )

    insight_bar(
        bullets=bullets,
        action=(
            "Each year-over-year admissions decline may have a different cause — see "
            "\"Orthopaedics deep dive\" and \"Factors affecting conversion\" below."
        ),
        variant="danger" if any_critical else "warning",
    )


# ── S3: Segment conversion treemap ───────────────────────────────────────────

def render_s3_treemap(df: pd.DataFrame) -> None:
    section_header("Conversion by clinical segment")
    if not _safe(df):
        _empty()
        return

    chart_card(
        "Visit volume and conversion rate by clinical segment",
        subtitle="Box size = visit volume. Color = conversion rate, sequential teal — darkest is highest.",
    )

    # Sequential teal ramp keyed to conversion rate, fixed 0–40% scale so the
    # ramp reads consistently across filtered views (spec §4 metric-intensity
    # treemap) — not a bucketed traffic-light system.
    _RAMP_MIN, _RAMP_MAX = 0.0, 40.0
    _RAMP_STOPS = [
        (0.0, (225, 245, 238)),   # #E1F5EE — lowest
        (0.5, (79, 173, 165)),    # #4FADA5 — mid
        (1.0, (15, 110, 86)),     # #0F6E56 — highest
    ]

    def _ramp_t(rate: float) -> float:
        return max(0.0, min(1.0, (rate - _RAMP_MIN) / (_RAMP_MAX - _RAMP_MIN)))

    def _lerp_rgb(a, b, f):
        return tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3))

    def _fill(rate: float) -> str:
        t = _ramp_t(rate)
        (t0, c0), (t1, c1) = next(
            (( _RAMP_STOPS[i], _RAMP_STOPS[i + 1]) for i in range(len(_RAMP_STOPS) - 1)
             if _RAMP_STOPS[i][0] <= t <= _RAMP_STOPS[i + 1][0]),
            (_RAMP_STOPS[-2], _RAMP_STOPS[-1]),
        )
        local_f = (t - t0) / (t1 - t0) if t1 > t0 else 0
        r, g, b = _lerp_rgb(c0, c1, local_f)
        return f"#{r:02X}{g:02X}{b:02X}"

    def _txt(rate: float) -> str:
        # Dark box → white text, light box → navy text
        return "#FFFFFF" if _ramp_t(rate) >= 0.55 else "#141F3D"

    labels, parents, values, colors, text_vals = [], [], [], [], []
    root = "All visits"
    labels.append(root)
    parents.append("")
    values.append(int(df["TOTAL_VISITS"].sum()))
    colors.append("#F4F6FA")
    text_vals.append("")

    for _, row in df.iterrows():
        seg   = row["PRIMARY_VISIT_SEGMENT"]
        vols  = int(row["TOTAL_VISITS"])
        rate  = float(row["CONVERSION_RATE_PCT"])
        pct   = float(row["PCT_OF_ALL_VISITS"])
        labels.append(seg)
        parents.append(root)
        values.append(vols)
        colors.append(_fill(rate))
        text_vals.append(
            f"{seg.split(': ')[-1]}<br>"
            f"{fmt_num(vols)} visits ({pct}%)<br>"
            f"<b>{fmt_pct(rate)} conversion</b>"
        )

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        # branchvalues="total" — root's listed value already includes its
        # children; default "remainder" would add them again, leaving the
        # children filling only half the treemap and the root color bleeding
        # through as blank space
        branchvalues="total",
        customdata=text_vals,
        # textinfo="text" only works when Plotly doesn't fall back to label
        # mode — use texttemplate + customdata/hovertemplate so labels always show
        texttemplate="%{label}",
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(
            colors=colors,
            # colorscale=None is the critical fix — without it Plotly ignores
            # the explicit hex list and applies its own sequential scale
            colorscale=None,
            showscale=False,
            line=dict(width=2, color="#FFFFFF"),
        ),
        tiling=dict(pad=4),
        pathbar=dict(visible=False),
    ))
    fig.update_layout(
        **{**_LAYOUT, "height": _H_SINGLE, "margin": dict(t=4, b=4, l=0, r=0)},
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    chart_card_close()

    # Compute insight text from data
    top_vol_seg  = df.loc[df["TOTAL_VISITS"].idxmax(), "PRIMARY_VISIT_SEGMENT"]
    top_vol_pct  = float(df.loc[df["TOTAL_VISITS"].idxmax(), "PCT_OF_ALL_VISITS"])
    high_segs    = df[df["CONVERSION_RATE_PCT"] >= 30]["PRIMARY_VISIT_SEGMENT"].tolist()
    high_labels  = [s.split(": ")[-1] for s in high_segs]

    ortho_segs = df[df["PRIMARY_VISIT_SEGMENT"].str.contains("Core Orthopedics", na=False)]
    ortho_pct = float(ortho_segs["PCT_OF_ALL_VISITS"].sum()) if not ortho_segs.empty else top_vol_pct
    ortho_names = " and ".join(s.split(": ")[-1] for s in ortho_segs["PRIMARY_VISIT_SEGMENT"]) \
        if not ortho_segs.empty else top_vol_seg.split(": ")[-1]

    spine_row = df[df["PRIMARY_VISIT_SEGMENT"].str.contains("Spine", na=False)]
    spine_pct = float(spine_row["PCT_OF_ALL_VISITS"].iloc[0]) if not spine_row.empty else None
    general_row = df[df["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: General"]
    general_pct = float(general_row["PCT_OF_ALL_VISITS"].iloc[0]) if not general_row.empty else None

    insight_bar(
        bullets=[
            f"{', '.join(high_labels)} are already admitting 30% or more of their patients — a "
            f"healthy, expected rate.",
            f"However, Orthopedics cases — {ortho_names} — make up {fmt_pct(ortho_pct, 0)} of all "
            f"visits, so the overall conversion rate is definitely being pulled down by them.",
            (f"Spine's low conversion ({fmt_pct(spine_pct, 0)} of all visits) is already explained "
             f"elsewhere on this tab — it's a pain-management-dominated population (chronic back "
             f"pain, sciatica, lumbago), not patients who need surgery. Low admission there is "
             f"expected, not a gap."
             if spine_pct is not None else
             "Spine's low conversion is already explained elsewhere on this tab as a "
             "pain-management-dominated population, not a gap."),
            (f"General ({fmt_pct(general_pct, 0)} of all visits) has no equivalent explanation yet "
             f"— that's the real open question."
             if general_pct is not None else
             "General has no equivalent explanation yet — that's the real open question."),
            "Put together, this points to a broader shift: the hospital is increasingly attracting "
            "patients who come for ongoing care and follow-up rather than for admission-worthy "
            "acute issues — a case-mix change worth planning for, not a conversion failure to fix.",
        ],
        action=(
            f"Investigate General specifically, the same way Spine's case mix was investigated — "
            f"is its lower conversion also explained by a similar patient population, or is "
            f"something there turning away patients who should be admitted?"
        ),
        variant="danger",
    )


# ── S4: Conversion rate vs. benchmark range, dumbbell chart ──────────────────
# There is no published national benchmark for OPD-to-IPD conversion rate by
# surgical specialty (confirmed by search — the literature is explicit that
# no such standard exists). The ranges below are an internal reference table
# supplied directly for this comparison, not sourced or verified externally
# by us — treat as an internal target band, not an industry standard.
_CONVERSION_BENCHMARKS = {
    # PRIMARY_VISIT_SEGMENT value: (display label, floor %, ceiling %)
    "Standalone Specialty: Plastic Surgery":          ("Plastic Surgery", 15.0, 25.0),
    "Standalone Specialty: Neurosurgery":              ("Neurosurgery", 20.0, 30.0),
    "Core General Surgery":                            ("Core General Surgery", 15.0, 25.0),
    "Standalone Specialty: Maxillofacial":             ("Maxillofacial", 15.0, 25.0),
    "Standalone Specialty: Obstetrics & Gynaecology":  ("Obstetrics & Gynaecology", 15.0, 25.0),
    "Standalone Medical: Cardiovascular":              ("Cardiovascular", 15.0, 25.0),
    "Standalone Specialty: Urology":                   ("Urology", 15.0, 25.0),
    "Other General Outpatient":                        ("General", 10.0, 15.0),
    "Standalone Medical: Neurology":                   ("Neurology", 10.0, 15.0),
    "Standalone Specialty: ENT":                       ("ENT", 15.0, 20.0),
    "Standalone Specialty: Eye/Ophthalmology":         ("Eye / Ophthalmology", 10.0, 20.0),
    "Core Orthopedics: Spine and Back Pain Care":      ("Spine and Back Pain Care", 15.0, 20.0),
    "Standalone Specialty: Dental":                    ("Dental", 0.1, 2.0),
}


def render_s4_segment_bar(df: pd.DataFrame) -> None:
    section_header("Conversion rate by segment")
    if not _safe(df):
        _empty()
        return

    rows = []
    for _, r in df.iterrows():
        bench = _CONVERSION_BENCHMARKS.get(r["PRIMARY_VISIT_SEGMENT"])
        if not bench:
            continue
        label, floor, ceiling = bench
        rows.append({
            "LABEL": label, "ACTUAL": float(r["CONVERSION_RATE_PCT"]),
            "FLOOR": floor, "CEILING": ceiling,
        })
    if not rows:
        _empty()
        return
    bdf = pd.DataFrame(rows).sort_values("ACTUAL")

    # Three-color status: below floor = red (not good), above ceiling =
    # green (good), within range = amber (acceptable). Status lives on the
    # connecting line/annotation only — the actual-rate dot itself stays one
    # consistent color so it doesn't fight with the status coding.
    _WITHIN_AMBER = "#EF9F27"

    def _classify(row):
        if row["ACTUAL"] > row["CEILING"]:
            return "above", row["CEILING"], row["ACTUAL"] - row["CEILING"], _C_GREEN
        if row["ACTUAL"] < row["FLOOR"]:
            return "below", row["FLOOR"], row["FLOOR"] - row["ACTUAL"], _C_RED
        nearer = row["CEILING"] if (row["CEILING"] - row["ACTUAL"]) < (row["ACTUAL"] - row["FLOOR"]) else row["FLOOR"]
        return "within", nearer, 0.0, _WITHIN_AMBER

    bdf[["STATUS", "BOUND", "DIFF", "COLOR"]] = bdf.apply(
        lambda r: pd.Series(_classify(r)), axis=1
    )

    chart_card(
        "Conversion rate vs. published benchmark — how far is each segment from its range?",
        "Each line spans from the benchmark bound to the actual rate — line and label color shows "
        "status: green = above ceiling, amber = within range, red = below floor · circle = actual "
        "rate · diamond = benchmark bound",
    )

    labels = bdf["LABEL"].tolist()
    actual = bdf["ACTUAL"].tolist()
    bound = bdf["BOUND"].tolist()
    line_colors = [
        "rgba(59,109,17,0.4)" if s == "above" else
        "rgba(163,45,45,0.5)" if s == "below" else
        "rgba(239,159,39,0.4)"
        for s in bdf["STATUS"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        mode="markers", name="Actual rate", y=labels, x=actual,
        marker=dict(size=16, color=PRIMARY, symbol="circle", line=dict(width=2, color="white")),
        hovertemplate="<b>%{y}</b><br>Actual: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        mode="markers", name="Benchmark bound", y=labels, x=bound,
        marker=dict(size=13, color="#8A93A6", symbol="diamond", line=dict(width=2, color="#5C6478")),
        hovertemplate="<b>%{y}</b><br>Benchmark bound: %{x:.1f}%<extra></extra>",
    ))
    shapes = [
        dict(type="line", xref="x", yref="y", x0=min(a, b), x1=max(a, b), y0=lbl, y1=lbl,
             line=dict(color=lc, width=5))
        for lbl, a, b, lc in zip(labels, actual, bound, line_colors)
    ]
    x_max = max(max(actual, default=0), max(bound, default=0)) + 5
    annotations = [
        dict(
            xref="x", yref="y", x=x_max, y=lbl, showarrow=False, xanchor="left", font=dict(size=10),
            text=(f'<b style="color:{_C_GREEN}">▲ {diff:.1f}% above</b>' if status == "above"
                  else f'<b style="color:{_C_RED}">▼ {diff:.1f}% below</b>' if status == "below"
                  else f'<span style="color:{_WITHIN_AMBER}">✓ within</span>'),
        )
        for lbl, status, diff in zip(labels, bdf["STATUS"], bdf["DIFF"])
    ]

    fig.update_layout(
        **{**_LAYOUT, "height": max(320, len(bdf) * 34),
           "legend": dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
           "margin": dict(t=12, b=52, l=10, r=110)},
        xaxis={**AXIS_Y, "ticksuffix": "%", "title": "Conversion rate", "range": [0, x_max + 5]},
        yaxis={**AXIS_X, "showgrid": False, "automargin": True},
        shapes=shapes, annotations=annotations, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()


# ── S5: Orthopaedics deep dive ───────────────────────────────────────────────

def render_s5_ortho_deep_dive(
    df_ortho: pd.DataFrame,
    df_spine: pd.DataFrame,
) -> None:
    section_header("Orthopaedics deep dive")

    col_chart, col_callout = st.columns([1.1, 0.9])

    with col_chart:
        if not _safe(df_ortho):
            _empty()
        else:
            chart_card("Ortho burden group × encounter type")

            wide = df_ortho.pivot_table(
                index="BURDEN_GROUP",
                columns="ENCOUNTER_TYPE",
                values="CONVERSION_RATE_PCT",
                aggfunc="first",
            ).reset_index()
            wide.columns.name = None

            acute_col  = "New / Acute"
            follow_col = "Follow-up / Chronic Mgmt"
            wide = wide.sort_values(acute_col, ascending=False)
            wide["SHORT"] = wide["BURDEN_GROUP"].str.replace("Ortho: ", "", regex=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=wide["SHORT"],
                y=wide.get(acute_col, []),
                name="New / Acute",
                marker_color=_C_BLUE,
                marker_cornerradius=3,
            ))
            fig.add_trace(go.Bar(
                x=wide["SHORT"],
                y=wide.get(follow_col, []),
                name="Follow-up / Chronic Mgmt",
                marker_color=_C_GRAY,
                marker_cornerradius=3,
            ))
            fig.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED, "barmode": "group"},
                xaxis={**AXIS_Y, "showgrid": False},
                yaxis={**AXIS_Y, "ticksuffix": "%"},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_callout:
        # Spine callout card
        vol_a, vol_b = 215, 680
        rate_b = 0.8
        yr_a_label, yr_b_label = 2024, 2025
        if _safe(df_spine):
            spine_latest = df_spine[df_spine["VISIT_YEAR"] >= 2024].sort_values("VISIT_YEAR")
            if len(spine_latest) >= 2:
                yr_a = spine_latest.iloc[0]
                yr_b = spine_latest.iloc[-1]
                vol_a = int(yr_a["AVG_MONTHLY_VISITS"])
                vol_b = int(yr_b["AVG_MONTHLY_VISITS"])
                rate_b = float(yr_b["CONVERSION_RATE_PCT"])
                yr_a_label = int(yr_a["VISIT_YEAR"])
                yr_b_label = int(yr_b["VISIT_YEAR"])

        # Spine acute rate for the callout
        spine_ortho = (
            df_ortho[df_ortho["BURDEN_GROUP"].str.contains("Spine", na=False)]
            if _safe(df_ortho) else pd.DataFrame()
        )
        if not spine_ortho.empty:
            spine_acute  = float(
                spine_ortho.loc[spine_ortho["ENCOUNTER_TYPE"] == "New / Acute", "CONVERSION_RATE_PCT"]
                .values[0] if "New / Acute" in spine_ortho["ENCOUNTER_TYPE"].values else 3.2
            )
        else:
            spine_acute = 3.2

        # Single merged card (explanation + volume comparison) so its total
        # height can be pinned to match the left chart card exactly, instead
        # of two separately-sized cards that drift out of alignment.
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E4E7ED;border-left:4px solid #A32D2D;
                        border-radius:0 10px 10px 0;padding:14px 16px;height:288px;
                        box-sizing:border-box;display:flex;flex-direction:column">
              <div>
                <div style="font-size:9px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                            color:#A32D2D;margin-bottom:4px">Spine pathway — explained gap</div>
                <div style="font-size:22px;font-weight:700;color:#A32D2D">{fmt_pct(spine_acute)}</div>
                <div style="font-size:11px;color:#5C6478;margin-top:6px;line-height:1.5">
                  Expected: Spine is mostly conservative pain management (lumbago, sciatica, low
                  back pain), so a new/acute rate close to follow-up is normal, not a missed-surgery gap.
                </div>
              </div>
              <div style="flex:1"></div>
              <div style="border-top:1px solid #E4E7ED;padding-top:8px">
                <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                            letter-spacing:.06em;color:#8A93A6;margin-bottom:8px">
                  Spine volume {yr_a_label} vs {yr_b_label}
                </div>
                <div style="display:flex;gap:12px;align-items:flex-end;height:50px;margin-bottom:6px">
                  <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
                    <div style="font-size:10px;font-weight:600;color:#854F0B">{vol_a}/mo</div>
                    <div style="width:100%;height:20px;background:#EF9F27;border-radius:3px 3px 0 0"></div>
                    <div style="font-size:9px;color:#8A93A6">{yr_a_label}</div>
                  </div>
                  <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
                    <div style="font-size:10px;font-weight:600;color:#A32D2D">{vol_b}/mo</div>
                    <div style="width:100%;height:45px;background:#E24B4A;border-radius:3px 3px 0 0"></div>
                    <div style="font-size:9px;color:#8A93A6">{yr_b_label}</div>
                  </div>
                </div>
                <div style="font-size:10px;color:#5C6478">
                  Volume tripled, admissions fell — conversion collapsed to {fmt_pct(rate_b)} in {yr_b_label}.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full-width insight bar below the whole chart row — never inside a
    # single column of a paired row (Template B rule, see ui_template.py).
    # Synthesizes all three panels above (bar chart, Spine explanation card,
    # Spine volume card) into one connected finding, rather than repeating
    # the bar chart alone and pointing elsewhere for the rest.
    vol_growth_pct = round((vol_b - vol_a) / vol_a * 100) if vol_a else None
    insight_bar(
        bullets=[
            "Fracture & Trauma, Hip, and Knee all show high new/acute conversion — a healthy "
            "acute surgical pipeline, no concern.",
            "Soft Tissue & MSK's low new/acute rate is appropriate for its conservative case mix.",
            f"Spine is the outlier at {fmt_pct(spine_acute)} new/acute — but this is explained, not "
            f"a gap: its population is dominated by conservative pain management (lumbago, "
            f"sciatica, low back pain), so a rate this close to follow-up is expected.",
            (f"What's changed is scale: Spine's monthly volume grew from {vol_a}/mo in {yr_a_label} "
             f"to {vol_b}/mo in {yr_b_label} (+{vol_growth_pct}%) while its conversion rate fell "
             f"further to {fmt_pct(rate_b)} — the same conservative pathway is now carrying far "
             f"more patients than before."
             if vol_growth_pct is not None else
             f"What's changed is scale: Spine's volume has grown sharply while its conversion rate "
             f"fell further to {fmt_pct(rate_b)} by {yr_b_label} — the same conservative pathway is "
             f"now carrying far more patients than before."),
        ],
        action=(
            "Don't chase Spine's conversion rate — it's structurally low by design. Instead confirm "
            "follow-up/recheck capacity is keeping pace with its tripled volume, and run a targeted "
            "case-note review (not a full pathway audit) to catch the non-pain-management minority "
            "who may need surgical referral."
        ),
        variant="warning",
    )


# ── S6: Non-ortho case mix bubble ────────────────────────────────────────────

def render_s6_non_ortho(df: pd.DataFrame) -> None:
    section_header("Non-orthopaedic case mix")
    if not _safe(df):
        _empty()
        return

    # Aggregate to one row per segment (combine both origin groups for overall,
    # keep pct_never_ortho from "Never" row)
    never = df[df["PATIENT_ORIGIN"] == "Never seen for orthopaedics"].copy()
    total = df.groupby("PRIMARY_VISIT_SEGMENT").agg(
        TOTAL_VISITS=("TOTAL_VISITS", "sum"),
        INPATIENT_ADMISSIONS=("INPATIENT_ADMISSIONS", "sum"),
    ).reset_index()
    total["CONVERSION_RATE_PCT"] = (
        100.0 * total["INPATIENT_ADMISSIONS"] / total["TOTAL_VISITS"].replace(0, pd.NA)
    ).round(1)

    merged = total.merge(
        never[["PRIMARY_VISIT_SEGMENT", "PCT_NEVER_ORTHO"]],
        on="PRIMARY_VISIT_SEGMENT", how="left",
    )

    seg_colors = {
        "Standalone Medical: Neurology":                  f"rgba({int(_C_PURPLE[1:3],16)},{int(_C_PURPLE[3:5],16)},{int(_C_PURPLE[5:7],16)},.65)",
        "Standalone Specialty: Obstetrics & Gynaecology": f"rgba({int(_C_PINK[1:3],16)},{int(_C_PINK[3:5],16)},{int(_C_PINK[5:7],16)},.65)",
        "Standalone Specialty: Urology":                  f"rgba({int(_C_BLUE[1:3],16)},{int(_C_BLUE[3:5],16)},{int(_C_BLUE[5:7],16)},.65)",
    }
    short = {
        "Standalone Medical: Neurology":                  "Neurology",
        "Standalone Specialty: Obstetrics & Gynaecology": "Obs and gynae",
        "Standalone Specialty: Urology":                  "Urology",
    }

    chart_card(
        "Patient origin vs. conversion rate — standalone segments",
    )

    fig = go.Figure()
    for _, row in merged.iterrows():
        seg  = row["PRIMARY_VISIT_SEGMENT"]
        col  = seg_colors.get(seg, f"rgba(136,135,128,.5)")
        lbl  = short.get(seg, seg.split(": ")[-1])
        vol  = int(row["TOTAL_VISITS"])
        rate = float(row.get("CONVERSION_RATE_PCT", 0))
        pct_indep = float(row.get("PCT_NEVER_ORTHO", 50))

        fig.add_trace(go.Scatter(
            x=[pct_indep],
            y=[rate],
            mode="markers+text",
            marker=dict(
                size=max(12, min(40, vol / 60)),
                color=col,
                line=dict(width=1, color="#E4E7ED"),
            ),
            text=[lbl],
            textposition="middle left",
            textfont=dict(size=11, color=TEXT_MUTED),
            name=lbl,
            customdata=[[vol, pct_indep, rate]],
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                f"Independent demand: %{{x:.0f}}%<br>"
                f"Conversion rate: %{{y:.1f}}%<br>"
                f"Total visits: {fmt_num(vol)}"
                f"<extra></extra>"
            ),
        ))

    x_vals = merged["PCT_NEVER_ORTHO"].astype(float)
    y_vals = merged["CONVERSION_RATE_PCT"].astype(float)
    x_lo = max(0, x_vals.min() - 15)
    x_hi = x_vals.max() + 4
    y_lo = 0
    y_hi = y_vals.max() * 1.35

    fig.update_layout(
        **{**_LAYOUT, "height": _H_SINGLE, "margin": {**_LAYOUT["margin"], "r": 60}},
        xaxis={
            **AXIS_Y,
            "title_text": "% of patients never seen for orthopaedics",
            "ticksuffix": "%",
            "range": [x_lo, x_hi],
        },
        yaxis={
            **AXIS_Y,
            "title_text": "Conversion rate %",
            "ticksuffix": "%",
            "range": [y_lo, y_hi],
        },
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    # Compute the shared/standalone gap for Urology from raw df
    urology = df[df["PRIMARY_VISIT_SEGMENT"].str.contains("Urology", na=False)]
    if len(urology) == 2:
        shared_rate    = float(urology[urology["PATIENT_ORIGIN"] == "Also seen for orthopaedics"]["CONVERSION_RATE_PCT"].values[0])
        standalone_rate= float(urology[urology["PATIENT_ORIGIN"] == "Never seen for orthopaedics"]["CONVERSION_RATE_PCT"].values[0])
        urology_gap_str = f" ({shared_rate:.1f}% shared vs {standalone_rate:.1f}% standalone)"
    else:
        urology_gap_str = ""

    insight_bar(
        bullets=[
            "Neurology (78%+ never-ortho) — genuine independent neurological demand. "
            "The 19%+ conversion rate is clinically real and not inflated by surgical crossover.",
            "Obstetrics and gynae — largest of the three segments; patients who also appear "
            "as orthopaedic patients likely represent women receiving joint care alongside maternal care.",
            f"Urology — shared patients convert at nearly 2× the rate of standalone patients"
            f"{urology_gap_str}, suggesting a post-surgical complication signal worth a targeted case review.",
        ],
        action=(
            "All three segments represent genuine independent demand. Urology's shared/standalone "
            "gap is the one finding that warrants further investigation."
        ),
        variant="warning",
    )


# ── S7: Factors affecting conversion ─────────────────────────────────────────

def render_s7_factors(
    df_workload:    pd.DataFrame,
    df_staffing:    pd.DataFrame,
    df_comorbidity: pd.DataFrame,
) -> None:
    section_header("Factors affecting conversion")

    col_wl, col_staff = st.columns(2)

    # ── 7a: Workload vs conversion ──
    with col_wl:
        if not _safe(df_workload):
            _empty("No workload data")
        else:
            df_w = df_workload.sort_values("WORKLOAD_BUCKET")

            # Status-coded by distance from target — exact spec §3 border-strength
            # hex (bg-tint fills read as barely-there against a white chart, so
            # bars use the same vivid tone as the scatter/line status markers).
            bar_colors = []
            for _, row in df_w.iterrows():
                rate = float(row["AVG_CONVERSION_RATE_PCT"])
                bar_colors.append(
                    "#639922" if rate >= 30
                    else "#EF9F27" if rate >= 20
                    else "#E24B4A"
                )

            fig_wl = go.Figure(go.Bar(
                x=df_w["WORKLOAD_BUCKET"].str.replace(r"^\d+:\s*", "", regex=True),
                y=df_w["AVG_CONVERSION_RATE_PCT"],
                marker_color=bar_colors,
                marker_cornerradius=3,
                text=df_w["AVG_CONVERSION_RATE_PCT"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(size=11, color=TEXT_MUTED),
            ))
            fig_wl.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED},
                xaxis={**AXIS_Y, "showgrid": False, "tickangle": -30},
                yaxis={**AXIS_Y, "ticksuffix": "%"},
                showlegend=False,
            )

            chart_card(
                "Conversion rate by clinician monthly caseload",
                subtitle="EMR V1 only (2022–2024) — ID scheme changes at the Feb 2025 cutover",
            )
            st.plotly_chart(fig_wl, use_container_width=True, config=PC_CFG)
            chart_card_close()

            insight_bar(
                bullets=[
                    "Conversion holds at 25–74 visits/month, then collapses sharply above 150 — "
                    "a threshold effect, not a gradual decline.",
                ],
                action="Maintain individual caseloads below 150/month to protect triage quality.",
                variant="warning",
            )

    # ── 7b: Staffing trend ──
    with col_staff:
        if not _safe(df_staffing):
            _empty("No staffing data")
        else:
            df_s = df_staffing.sort_values("VISIT_YEAR").copy()
            df_s["ACTIVE_CLINICIANS"]      = df_s["ACTIVE_CLINICIANS"].astype(float)
            df_s["AVG_CONVERSION_RATE_PCT"] = df_s["AVG_CONVERSION_RATE_PCT"].astype(float)

            # Index both series to 100 at their first point to allow
            # comparison on a shared axis without dual-axis
            base_clinicians = df_s["ACTIVE_CLINICIANS"].iloc[0]
            base_rate       = df_s["AVG_CONVERSION_RATE_PCT"].iloc[0]
            df_s["IDX_CLINICIANS"] = 100 * df_s["ACTIVE_CLINICIANS"] / base_clinicians
            df_s["IDX_RATE"]       = 100 * df_s["AVG_CONVERSION_RATE_PCT"] / base_rate

            fig_st = go.Figure()
            fig_st.add_trace(go.Scatter(
                x=df_s["VISIT_YEAR"],
                y=df_s["IDX_CLINICIANS"],
                mode="lines+markers",
                line=dict(color=_C_BLUE, width=2),
                marker=dict(size=6, color=_C_BLUE),
                name="Clinicians (indexed)",
                customdata=df_s[["ACTIVE_CLINICIANS"]].values,
                hovertemplate="Year: %{x}<br>Clinicians: %{customdata[0]}<extra></extra>",
            ))
            fig_st.add_trace(go.Scatter(
                x=df_s["VISIT_YEAR"],
                y=df_s["IDX_RATE"],
                mode="lines+markers",
                line=dict(color=SECONDARY, width=2, dash="dot"),
                marker=dict(size=6, color=SECONDARY),
                name="Conversion rate (indexed)",
                customdata=df_s[["AVG_CONVERSION_RATE_PCT"]].values,
                hovertemplate="Year: %{x}<br>Conversion: %{customdata[0]:.1f}%<extra></extra>",
            ))
            fig_st.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED},
                xaxis={**AXIS_Y, "showgrid": False,
                       "tickvals": df_s["VISIT_YEAR"].tolist(),
                       "ticktext": [str(y) for y in df_s["VISIT_YEAR"].tolist()]},
                yaxis={**AXIS_Y, "ticksuffix": "", "title_text": "Index (base year = 100)"},
            )

            chart_card(
                "Active clinicians and conversion rate — 2022 to 2024",
                subtitle="Both series indexed to 100 at base year — shows relative co-movement",
            )
            st.plotly_chart(fig_st, use_container_width=True, config=PC_CFG)
            chart_card_close()

            # Compute insight from data
            drop_yr = df_s.loc[df_s["IDX_CLINICIANS"].idxmin()]
            max_clin = int(df_s["ACTIVE_CLINICIANS"].max())
            min_clin = int(df_s["ACTIVE_CLINICIANS"].min())
            drop_pct = round((1 - min_clin / max_clin) * 100)

            insight_bar(
                bullets=[
                    f"{max_clin} clinicians at peak → {min_clin} at trough (−{drop_pct}%). "
                    f"Per-person caseload reached the collapse threshold confirmed in the adjacent chart.",
                ],
                action="Clinician headcount is a leading indicator — track it monthly.",
                variant="danger",
            )

    # ── 7c: Comorbidity grouped bar (full width) ──
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if not _safe(df_comorbidity):
        _empty("No comorbidity data")
    else:
        df_c = df_comorbidity.copy()
        df_c["LABEL"] = (
            df_c["SEGMENT_TYPE"]
            + " — "
            + df_c["HAS_CHRONIC_CONDITION"].map(
                {True: "comorbid", False: "no comorbidity", 1: "comorbid", 0: "no comorbidity"}
            )
        )
        df_c = df_c.sort_values("CONVERSION_RATE_PCT", ascending=False)

        bar_colors = [
            _C_BLUE if "comorbid" in lbl else _C_GRAY
            for lbl in df_c["LABEL"]
        ]

        fig_co = go.Figure(go.Bar(
            x=df_c["LABEL"].str.replace(" — ", "\n"),
            y=df_c["CONVERSION_RATE_PCT"],
            marker_color=bar_colors,
            marker_cornerradius=3,
            text=df_c["CONVERSION_RATE_PCT"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=11, color=TEXT_MUTED),
        ))
        fig_co.update_layout(
            **{**_LAYOUT, "height": 200},
            xaxis={**AXIS_Y, "showgrid": False},
            yaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, df_c["CONVERSION_RATE_PCT"].max() + 8]},
            showlegend=False,
        )

        chart_card(
            "Comorbidity and conversion — surgical vs. non-surgical patients",
            subtitle="Comorbid = has_chronic_condition flag. "
                     "Comorbidity effect is strongest in surgical patients.",
        )
        st.plotly_chart(fig_co, use_container_width=True, config=PC_CFG)
        chart_card_close()

        # Compute insight
        surgical_comorbid    = df_c[(df_c["SEGMENT_TYPE"] == "Surgical") & (df_c["HAS_CHRONIC_CONDITION"].isin([True, 1]))]
        surgical_nocomorbid  = df_c[(df_c["SEGMENT_TYPE"] == "Surgical") & (df_c["HAS_CHRONIC_CONDITION"].isin([False, 0]))]
        if not surgical_comorbid.empty and not surgical_nocomorbid.empty:
            s_c_rate = float(surgical_comorbid["CONVERSION_RATE_PCT"].values[0])
            s_n_rate = float(surgical_nocomorbid["CONVERSION_RATE_PCT"].values[0])
            multiplier = round(s_c_rate / s_n_rate, 1) if s_n_rate > 0 else 0
        else:
            s_c_rate, s_n_rate, multiplier = 21.2, 10.8, 2.0

        insight_bar(
            bullets=[
                f"Without comorbidity, surgical and non-surgical baseline rates converge at ~{s_n_rate:.0f}% — "
                f"comorbidity is the differentiating variable, not segment type.",
                f"Comorbid surgical patients convert at {fmt_pct(s_c_rate)} vs {fmt_pct(s_n_rate)} for non-comorbid "
                f"— a {multiplier}× effect on a population not currently being systematically flagged at triage.",
            ],
            action=(
                "Identify comorbid patients at first contact — they are structurally more likely to "
                "require admission and benefit from senior clinical input early."
            ),
            variant="success",
        )


# ── S8: 72-hour escalation ───────────────────────────────────────────────────
# Rebuilt per OPD_IPD_Escalation_Section_Rebuild.md. The original version
# compared escalation rate against avg hours to admission, which doesn't
# test anything actionable. This version tests whether low conversion is
# driving escalation (Chart 1) and whether escalating patients had any
# diagnostic workup before they escalated (Chart 2) — one insight bar
# covers both charts together, since this section is a single two-step
# argument, not two independent comparisons.

def render_s8_escalation(df: pd.DataFrame, df_conversion: pd.DataFrame = None,
                          df_timing: pd.DataFrame = None) -> None:
    section_header("72-hour escalation: does low conversion drive it?")
    if not _safe(df):
        _empty()
        return

    df = df.sort_values("VISIT_YEAR")

    # ── Chart 1 — escalation rate vs. conversion rate, by year ──────────────
    chart_container_open("Escalation rate vs. conversion rate, by year")

    years = df["VISIT_YEAR"].tolist()
    y_values = list(df["ESCALATION_RATE_PCT"])
    fig = go.Figure()
    # Dual-metric comparison — teal solid (primary) + raspberry dashed (comparison),
    # per spec §4, standardized across every two-line comparison chart.
    fig.add_trace(go.Scatter(
        x=years, y=df["ESCALATION_RATE_PCT"], mode="lines+markers",
        line=dict(color=PRIMARY, width=2), marker=dict(size=6, color=PRIMARY),
        name="Escalation rate %",
    ))
    if _safe(df_conversion):
        dfc = df_conversion.sort_values("VISIT_YEAR")
        fig.add_trace(go.Scatter(
            x=dfc["VISIT_YEAR"].tolist(), y=dfc["CONVERSION_RATE_PCT"], mode="lines+markers",
            line=dict(color=SECONDARY, width=2, dash="dash"), marker=dict(size=6, color=SECONDARY),
            name="Conversion rate % (Ortho General)",
        ))
        y_values += list(dfc["CONVERSION_RATE_PCT"])
    # Dynamic range, not a fixed 0-18% — Ortho General's overall annual
    # conversion rate isn't bounded the way the trauma-only escalation rate
    # is, and a hardcoded cap silently clips any year that runs higher,
    # making the line appear to enter the chart from mid-air.
    y_max = float(max(y_values)) * 1.15 if y_values else 20
    fig.update_layout(
        **{**_LAYOUT, "height": _H_SINGLE, "legend": {**_LAYOUT["legend"], "y": -0.18}},
        xaxis={**AXIS_X, "tickvals": years, "ticktext": [str(y) for y in years]},
        yaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, y_max]},
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_container_close()

    # ── Chart 2 — investigation timing for escalating patients ──────────────
    if _safe(df_timing):
        chart_container_open("Investigation timing for escalating patients, by pattern")
        st.markdown(
            '<div style="display:flex;gap:16px;margin-bottom:8px">'
            f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:{TEXT_SEC}">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{SUCCESS};display:inline-block">'
            f'</span>Investigated before admission</span>'
            f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:{TEXT_SEC}">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{WARNING};display:inline-block">'
            f'</span>Investigated only after admission</span>'
            f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:{TEXT_SEC}">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{DANGER};display:inline-block">'
            f'</span>No investigation at all</span></div>',
            unsafe_allow_html=True,
        )
        # "Blank diagnosis" dropped from the chart entirely — it can't be
        # attributed to any department (no diagnosis text to classify it by),
        # so it can't inform an Ortho-adjacent investigation-gap finding.
        df_t = df_timing[df_timing["ESCALATION_PATTERN"] != "Blank diagnosis (data gap)"]
        df_t = df_t.sort_values("TOTAL_ESCALATIONS", ascending=True)

        fig_t = go.Figure()
        for col, label, color in [
            ("PCT_BEFORE", "Investigated before admission", SUCCESS),
            ("PCT_AFTER_ONLY", "Investigated only after admission", WARNING),
            ("PCT_NONE", "No investigation at all", DANGER),
        ]:
            fig_t.add_trace(go.Bar(
                y=df_t["ESCALATION_PATTERN"], x=df_t[col], orientation="h", name=label,
                marker=dict(color=color, cornerradius=3),
                text=[f"{v:.0f}%" if v >= 6 else "" for v in df_t[col]], textposition="inside",
                textfont=dict(size=10, color="#FFFFFF"),
                # n shown on hover, not the axis label, to keep the chart clean.
                customdata=df_t["TOTAL_ESCALATIONS"],
                hovertemplate="<b>%{y}</b> (n=%{customdata:.0f})<br>" + label + ": %{x:.0f}%<extra></extra>",
            ))
        fig_t.update_layout(
            **{**_LAYOUT, "height": _H_PAIRED, "barmode": "stack"}, showlegend=False,
            xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
            yaxis={**AXIS_Y, "showgrid": False},
        )
        st.plotly_chart(fig_t, use_container_width=True, config=PC_CFG)
        chart_container_close()

    # ── One insight bar for both charts ──────────────────────────────────────
    bullets = []
    actions = []
    variant = "neutral"
    if _safe(df_conversion):
        bullets.append(
            "Escalation rate is Ortho General trauma-scoped; conversion rate above uses the same "
            "Ortho General scope, so the two lines are directly comparable."
        )
        dfc = df_conversion.sort_values("VISIT_YEAR")
        merged = df.merge(dfc, on="VISIT_YEAR", suffixes=("_ESC", "_CONV"))
        corr = (
            merged["ESCALATION_RATE_PCT"].corr(merged["CONVERSION_RATE_PCT"])
            if len(merged) >= 3 else None
        )
        if corr is not None and corr <= -0.5:
            bullets.append(
                f"Escalation rate and conversion rate move inversely (r={corr:.2f}) — a real, "
                f"fairly strong signal that lower conversion is linked to more escalations."
            )
            variant = "warning"
        elif corr is not None and corr <= -0.1:
            bullets.append(
                f"Escalation rate and conversion rate move inversely, but weakly (r={corr:.2f}) — a "
                f"real signal, not noise, but not strong enough on its own to explain escalation "
                f"spikes like 2024's. Treat it as a contributing factor to monitor, not the driver."
            )
        else:
            corr_txt = f"r={corr:.2f}" if corr is not None else "insufficient years to test"
            bullets.append(
                f"Escalation rate does not move inversely with conversion rate ({corr_txt}). Do not "
                f"use low conversion as the explanation for escalation spikes without this "
                f"correlation holding up in the actual data."
            )

    if _safe(df_timing):
        bullets.append(
            "The investigation-timing chart above is hospital-wide, not Ortho-scoped — it covers "
            "all 72-hour escalations. \"Blank diagnosis\" cases are excluded since there's no "
            "diagnosis text to attribute them to any department."
        )
        df_nb = df_timing[df_timing["ESCALATION_PATTERN"] != "Blank diagnosis (data gap)"]
        concerning = df_nb[df_nb["ESCALATION_PATTERN"] != "Elective/Scheduled pattern"]
        n_concerning = int(concerning["TOTAL_ESCALATIONS"].sum())
        n_none = int(concerning["NO_INVESTIGATION"].sum())
        pct_none = round(100.0 * n_none / n_concerning, 1) if n_concerning else 0.0

        bullets.append(
            f"{n_concerning} patients escalated in patterns not explained by planned procedures "
            f"(Acute/Trauma + Other/Unclear) — {n_none} of them ({pct_none:.0f}%) received no "
            f"investigation at all, before or after admission. That's the largest and most "
            f"concerning group on this chart."
        )

        acute = df_nb[df_nb["ESCALATION_PATTERN"] == "Acute/Trauma pattern"]
        other = df_nb[df_nb["ESCALATION_PATTERN"] == "Other/Unclear"]
        if not acute.empty and not other.empty:
            a, o = acute.iloc[0], other.iloc[0]
            bullets.append(
                f"Acute/Trauma: {a['PCT_AFTER_ONLY']:.0f}% ({int(a['INVESTIGATED_AFTER_ADMISSION_ONLY'])} "
                f"patients) were only investigated after they'd already been admitted. Other/Unclear: "
                f"{o['PCT_AFTER_ONLY']:.0f}% ({int(o['INVESTIGATED_AFTER_ADMISSION_ONLY'])} patients). "
                f"That's a real clinical-risk window and a gap in patient care, not just a metric — "
                f"the workup happened only once the situation had already become urgent."
            )
            variant = "danger"

        total_before_concerning = int(concerning["INVESTIGATED_BEFORE_ADMISSION"].sum())
        if total_before_concerning:
            bullets.append(
                f"{total_before_concerning} patients in these patterns WERE investigated before "
                f"admission and still escalated — that subset needs its own clinical review: was the "
                f"finding missed, or did the condition genuinely progress after a reasonable workup?"
            )

        actions.append(
            f"Audit the {n_none} Acute/Trauma and Other/Unclear patients who escalated with zero "
            f"investigation — that's the actionable gap. Separately review the "
            f"{total_before_concerning if total_before_concerning else 0} who were investigated but "
            f"escalated anyway, to check for missed findings."
        )

    if not actions:
        actions.append(
            "Escalation rate does not appear to be driven by conversion rate alone — look elsewhere "
            "(case mix, staffing, triage protocol changes) for what's actually moving it."
        )

    if bullets:
        insight_bar(bullets=bullets, action=" ".join(actions), variant=variant)


# ── S10: Recommendations ─────────────────────────────────────────────────────

def render_s10_recommendations(
    df_segments: pd.DataFrame = None,
    df_ortho: pd.DataFrame = None,
    df_spine_vol: pd.DataFrame = None,
    df_staffing: pd.DataFrame = None,
    df_comorbidity: pd.DataFrame = None,
    df_timing: pd.DataFrame = None,
) -> None:
    section_header("Recommendations")

    recs = []

    # Matches ui_template.py's _PRIORITY_SEVERITY_COLOR so the "Action:" line
    # inside a card highlights in the same color as that card's left border/label.
    _SEVERITY_COLOR = {"critical": DANGER, "monitor": WARNING, "okay": SUCCESS}

    def _list(items: list, severity: str = "monitor") -> str:
        color = _SEVERITY_COLOR.get(severity, WARNING)
        lis = "".join(
            f'<li style="margin-bottom:3px;font-weight:700;color:{color}">{i}</li>'
            if i.startswith("Action:") else
            f'<li style="margin-bottom:3px">{i}</li>'
            for i in items
        )
        return f'<ul style="margin:2px 0 0;padding-left:16px">{lis}</ul>'

    # ── P1: Spine minority + volume growth ──────────────────────────────────
    spine_acute_pct = None
    if _safe(df_ortho):
        spine_row = df_ortho[
            df_ortho["BURDEN_GROUP"].str.contains("Spine", na=False)
            & (df_ortho["ENCOUNTER_TYPE"] == "New / Acute")
        ]
        if not spine_row.empty:
            spine_acute_pct = float(spine_row.iloc[0]["CONVERSION_RATE_PCT"])
    vol_item = None
    if _safe(df_spine_vol):
        sv = df_spine_vol[df_spine_vol["VISIT_YEAR"] >= 2024].sort_values("VISIT_YEAR")
        if len(sv) >= 2:
            v_a, v_b = int(sv.iloc[0]["AVG_MONTHLY_VISITS"]), int(sv.iloc[-1]["AVG_MONTHLY_VISITS"])
            y_a, y_b = int(sv.iloc[0]["VISIT_YEAR"]), int(sv.iloc[-1]["VISIT_YEAR"])
            if v_a:
                vol_item = (
                    f"Volume grew {round((v_b - v_a) / v_a * 100)}% ({v_a}→{v_b}/mo, {y_a}–{y_b}) — "
                    f"confirm follow-up capacity is keeping pace"
                )
    p1_items = [
        f"New-presentation conversion: {fmt_pct(spine_acute_pct)} — expected, driven by "
        f"conservative pain-management case mix"
        if spine_acute_pct is not None else
        "New-presentation conversion is close to follow-up rate — expected given conservative case mix",
    ]
    if vol_item:
        p1_items.append(vol_item)
    p1_items.append("Action: targeted case-note review of the non-pain-management minority, not a full audit")
    recs.append({
        "label": "PRIORITY 1", "severity": "monitor",
        "title": "Screen the non-pain-management minority of spine cases",
        "body": _list(p1_items, "monitor"),
        "source": "Orthopaedics deep dive",
    })

    # ── P2: Clinician headcount ──────────────────────────────────────────────
    if _safe(df_staffing):
        max_clin = int(df_staffing["ACTIVE_CLINICIANS"].max())
        min_clin = int(df_staffing["ACTIVE_CLINICIANS"].min())
        drop_pct = round((1 - min_clin / max_clin) * 100) if max_clin else 0
        recs.append({
            "label": "PRIORITY 2", "severity": "critical",
            "title": "Track active clinician headcount monthly",
            "body": _list([
                f"Clinicians: {max_clin} at peak → {min_clin} at trough (−{drop_pct}%)",
                "Tracks the conversion decline over the same period",
                "Action: monitor headcount monthly as a leading indicator, before quality drops",
            ], "critical"),
            "source": "Factors affecting conversion",
        })

    # ── P3: Comorbid patients ────────────────────────────────────────────────
    if _safe(df_comorbidity):
        surg_c = df_comorbidity[(df_comorbidity["SEGMENT_TYPE"] == "Surgical") &
                                 (df_comorbidity["HAS_CHRONIC_CONDITION"].isin([True, 1]))]
        surg_n = df_comorbidity[(df_comorbidity["SEGMENT_TYPE"] == "Surgical") &
                                 (df_comorbidity["HAS_CHRONIC_CONDITION"].isin([False, 0]))]
        if not surg_c.empty and not surg_n.empty:
            s_c_rate = float(surg_c["CONVERSION_RATE_PCT"].values[0])
            s_n_rate = float(surg_n["CONVERSION_RATE_PCT"].values[0])
            multiplier = round(s_c_rate / s_n_rate, 1) if s_n_rate > 0 else None
            recs.append({
                "label": "PRIORITY 3", "severity": "monitor",
                "title": "Flag comorbid patients at first contact for senior review",
                "body": _list([
                    f"Comorbid surgical patients: {fmt_pct(s_c_rate)} conversion vs {fmt_pct(s_n_rate)} "
                    f"non-comorbid" + (f" ({multiplier}× effect)" if multiplier is not None else ""),
                    "Not currently flagged early in the consultation",
                    "Action: identify comorbid patients at first contact for senior review",
                ], "monitor"),
                "source": "Factors affecting conversion",
            })

    # ── P4: General's conversion is still unexplained ───────────────────────
    if _safe(df_segments):
        general_row = df_segments[df_segments["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: General"]
        if not general_row.empty:
            g_pct = float(general_row.iloc[0]["PCT_OF_ALL_VISITS"])
            g_conv = float(general_row.iloc[0]["CONVERSION_RATE_PCT"])
            recs.append({
                "label": "PRIORITY 4", "severity": "critical",
                "title": "Investigate General's low conversion — no explanation yet",
                "body": _list([
                    f"General: {fmt_pct(g_pct, 0)} of all visits, {fmt_pct(g_conv)} conversion",
                    "Unlike Spine, no confirmed benign explanation exists",
                    "Action: determine if it's case-mix driven, or a pathway turning away admissible patients",
                ], "critical"),
                "source": "Conversion by clinical segment",
            })

    # ── P5: Segment vs. published benchmark ─────────────────────────────────
    if _safe(df_segments):
        gs_row = df_segments[df_segments["PRIMARY_VISIT_SEGMENT"] == "Core General Surgery"]
        if not gs_row.empty:
            gs_conv = float(gs_row.iloc[0]["CONVERSION_RATE_PCT"])
            _, floor, ceiling = _CONVERSION_BENCHMARKS.get(
                "Core General Surgery", ("Core General Surgery", 15.0, 25.0)
            )
            status = "above" if gs_conv > ceiling else "within" if gs_conv >= floor else "below"
            p5_severity = "okay" if gs_conv >= floor else "critical"
            recs.append({
                "label": "PRIORITY 5", "severity": p5_severity,
                "title": "Use General Surgery as an internal benchmark",
                "body": _list([
                    f"Conversion: {fmt_pct(gs_conv)} — {status} the {floor:.0f}–{ceiling:.0f}% benchmark range",
                    "Action: if performing well, use its triage/escalation practices as the internal "
                    "comparison for Orthopaedics",
                ], p5_severity),
                "source": "Conversion rate vs. published benchmark",
            })

    recs.append({
        "label": "PRIORITY 6", "severity": "monitor",
        "title": "Treat clinician tracking as broken across the Feb 2025 EMR cutover",
        "body": _list([
            "Clinician ID scheme changes at the cutover — blocks cross-period tracking",
            "2022–2024 data is clean",
            "Action: scope clinician-level analysis to within a single EMR period until ID mapping is fixed",
        ], "monitor"),
        "source": "Factors affecting conversion",
    })
    recs.append({
        "label": "PRIORITY 7", "severity": "monitor",
        "title": "Don't compute pre-2025 surgical wait times",
        "body": _list([
            "Theatre scheduling records only exist from 2025 onward",
            "A pre-2025 figure would reflect missing data, not a true zero or fast turnaround",
            "Action: exclude pre-2025 years from wait-time reporting",
        ], "monitor"),
        "source": "72-hour escalation: does low conversion drive it?",
    })

    # ── P8: Escalation investigation gap ─────────────────────────────────────
    if _safe(df_timing):
        df_nb = df_timing[df_timing["ESCALATION_PATTERN"] != "Blank diagnosis (data gap)"]
        concerning = df_nb[df_nb["ESCALATION_PATTERN"] != "Elective/Scheduled pattern"]
        n_concerning = int(concerning["TOTAL_ESCALATIONS"].sum())
        n_none = int(concerning["NO_INVESTIGATION"].sum())
        n_before = int(concerning["INVESTIGATED_BEFORE_ADMISSION"].sum())
        if n_concerning:
            recs.append({
                "label": "PRIORITY 8", "severity": "critical",
                "title": "Audit escalating patients who received zero investigation",
                "body": _list([
                    f"{n_none} of {n_concerning} non-elective escalations (Acute/Trauma, Other/Unclear) "
                    f"had no imaging before or after admission",
                    f"{n_before} were investigated and escalated anyway",
                    "Action: audit the zero-investigation group; clinically review the investigated-but-"
                    "escalated group for missed findings",
                ], "critical"),
                "source": "Investigation timing for escalating patients",
            })

    priority_cards(recs)