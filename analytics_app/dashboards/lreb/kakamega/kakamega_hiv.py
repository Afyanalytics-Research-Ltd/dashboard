"""
Kakamega HIV Testing Services (HTS) — analysis dashboard.
"""
# 1. IMPORTS
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv

# 2. CONNECTION  
load_dotenv()

TABLE = "hospitals.staging.HIV_HTS_STAGING"

REQUIRED_ENV = [
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
]


@st.cache_resource(ttl=3000)
def get_connection():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required .env variable(s): {', '.join(missing)}")

    connect_kwargs = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )
    if os.getenv("SNOWFLAKE_ROLE"):
        connect_kwargs["role"] = os.environ["SNOWFLAKE_ROLE"]

    if os.getenv("SNOWFLAKE_AUTHENTICATOR"):
        connect_kwargs["authenticator"] = os.environ["SNOWFLAKE_AUTHENTICATOR"]
    elif os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"):
        connect_kwargs["private_key"] = _load_private_key(
            os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"],
            os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        )
    elif os.getenv("SNOWFLAKE_PASSWORD"):
        connect_kwargs["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    else:
        raise RuntimeError(
            "Set one of SNOWFLAKE_PRIVATE_KEY_PATH, SNOWFLAKE_PASSWORD, or "
            "SNOWFLAKE_AUTHENTICATOR in your .env file."
        )

    return snowflake.connector.connect(**connect_kwargs)


def _load_private_key(key_path, passphrase=None):
    """Read a PEM private key file (key-pair auth) and return it as the DER
    bytes the Snowflake connector's `private_key` kwarg expects."""
    from cryptography.hazmat.primitives import serialization

    with open(key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(),
            password=passphrase.encode() if passphrase else None,
        )
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _coerce_decimals(df):
    """Snowflake's fetch_pandas_all() returns ROUND()/NUMBER columns as
    decimal.Decimal objects (object dtype) rather than float64, which breaks
    plain arithmetic like `series * 2.5` downstream. Cast any such column to
    float so every numeric column behaves like a normal float column."""
    import decimal
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if not sample.empty and isinstance(sample.iloc[0], decimal.Decimal):
                df[col] = df[col].astype(float)
    return df


def _is_expired_auth_error(e):
    """True for Snowflake's 'Authentication token has expired' error (390114) —
    the connection object we cached via st.cache_resource is still sitting there,
    but the token it holds no longer works, so every query through it fails the
    same way until a fresh connection is made. Matched on message text (covers
    externalbrowser/OAuth token expiry and similar re-auth-required errors)
    rather than a specific exception class, since the connector can surface this
    as more than one error type."""
    msg = str(e)
    return "390114" in msg or "authentication token has expired" in msg.lower()


def _execute_sql(sql):
    """Runs one query against a fresh cursor, retrying exactly once — with the
    cached connection torn down and re-established — if the failure was an
    expired auth token. Without this, a mid-session token expiry (common with
    SSO/externalbrowser auth) would wedge every chart on the dashboard until
    someone manually restarted the app, since st.cache_resource has no way to
    know the connection it's holding has gone bad on its own."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            return _coerce_decimals(cur.fetch_pandas_all())
    except Exception as e:  # noqa: BLE001
        if not _is_expired_auth_error(e):
            raise
        get_connection.clear()
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            return _coerce_decimals(cur.fetch_pandas_all())


@st.cache_data(ttl=300)
def _run_fast(sql):
    return _execute_sql(sql)


@st.cache_data(ttl=1800)
def _run_slow(sql):
    return _execute_sql(sql)


def run_query(sql, speed="slow"):
    """speed='fast' -> 5 min cache (small lookups); 'slow' -> 30 min cache (full-table rollups)."""
    return _run_fast(sql) if speed == "fast" else _run_slow(sql)

# 3. STYLE / THEME  
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = "#2a78d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

NAVY = "#0b2545"
ACCENT = CATEGORICAL[2]          
ACCENT_DARK = "#158f66"
INK = {"primary": "#132436", "secondary": "#4b5b6b", "muted": "#8a97a3"}
SURFACE = "#ffffff"
PAGE_BG = "#f7f9fb"
GRIDLINE = "#e3e8ec"

SECTIONS = [
    "Overview", "Testing & Care Cascade", "Clinical & Data Quality", "Provider Performance",
]

st.set_page_config(page_title="Kakamega HIV HTS Dashboard", layout="wide", page_icon="📊")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif;
        color: {INK['primary']};
    }}
    h1, h2, h3, .app-title {{
        font-family: 'Poppins', system-ui, sans-serif !important;
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    .stApp {{ background-color: {PAGE_BG}; }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {GRIDLINE};
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        border-radius: 999px;
        padding: 6px 14px;
        margin-bottom: 2px;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background-color: {GRIDLINE};
    }}
    .stButton > button {{
        background-color: {ACCENT};
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_DARK};
        color: #ffffff;
    }}
    .kpi-card {{
        background-color: {SURFACE};
        border: 1px solid {GRIDLINE};
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 116px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .kpi-card .kpi-label {{ font-size: 13px; font-weight: 600; color: {INK['secondary']}; }}
    .kpi-card .kpi-value {{
        font-family: 'Poppins', system-ui, sans-serif;
        font-size: 26px; font-weight: 700; line-height: 1.15;
    }}
    .kpi-card .kpi-help {{ font-size: 12px; color: {INK['muted']}; }}
    .flag-box {{
        background-color: #fce8e6; border: 1px solid {STATUS['critical']}33;
        border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
    }}
    .flag-box b {{ color: {STATUS['critical']}; }}
    .flag-box p {{ color: {STATUS['critical']}; margin: 4px 0 0 0; }}
    .insight-note {{
        font-size: 13.5px;
        line-height: 1.5;
        color: {INK['secondary']};
        background-color: #f1f6f4;
        border-left: 3px solid {ACCENT};
        border-radius: 4px;
        padding: 8px 12px;
        margin: -4px 0 16px 0;
    }}
    .insight-note b {{ color: {INK['primary']}; }}
    .recommend-note {{
        font-size: 13.5px;
        line-height: 1.5;
        color: {INK['primary']};
        background-color: #eaf1fb;
        border-left: 3px solid {NAVY};
        border-radius: 4px;
        padding: 8px 12px;
        margin: -4px 0 16px 0;
    }}
    .recommend-note b {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_template(fig, y_title=None):
    """Apply the shared chart chrome: surface, gridlines, ink, no dual axis."""
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK["primary"], family="Inter, system-ui, -apple-system, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=64, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, linecolor=GRIDLINE, tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(gridcolor=GRIDLINE, linecolor=GRIDLINE, tickfont=dict(color=INK["muted"]), title=y_title)
    return fig


def status_color(value, good_at, warn_at, higher_is_better=True):
    if value is None or pd.isna(value):
        return INK["muted"]
    if higher_is_better:
        if value >= good_at:
            return STATUS["good"]
        if value >= warn_at:
            return STATUS["warning"]
        return STATUS["critical"]
    else:
        if value <= good_at:
            return STATUS["good"]
        if value <= warn_at:
            return STATUS["warning"]
        return STATUS["critical"]


def kpi_tile(col, label, value, help_text=None, color=None, small=False, min_height=None):
    """Fixed-height metric card — every card renders the same label / value /
    help structure so a row of cards always lines up, regardless of how long
    any individual value or help string is. Pass small=True for a value
    string too long to read comfortably at the default 26px (e.g. a date
    range) — the card stays the same height either way. Pass min_height to
    override the default 116px — e.g. to match a taller card sharing its row
    (a gauge_meter chart) so the row doesn't look uneven."""
    value_size = "16px" if small else "26px"
    height_style = f" min-height:{min_height}px;" if min_height else ""
    with col:
        st.markdown(
            f"""
            <div class="kpi-card" style="{height_style}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color or INK['primary']}; font-size:{value_size};">{value}</div>
              <div class="kpi-help">{help_text or "&nbsp;"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def bar_chart(df, x, y, title, color=SEQUENTIAL_BLUE, horizontal=False, pct=False, height=380, x_title=None):
    fig = go.Figure()
    if horizontal:
        fig.add_bar(y=df[x], x=df[y], orientation="h", marker_color=color)
        fig.update_yaxes(autorange="reversed")
    else:
        fig.add_bar(x=df[x], y=df[y], marker_color=color)
    fig.update_layout(title=title, height=height)
    fig = plotly_template(fig, y_title="%" if pct else None)
    if horizontal and x_title:
        fig.update_xaxes(title=x_title)
    st.plotly_chart(fig, width="stretch")


def volume_and_share_pair(df, dimension_label, color, horizontal=False, height=None, always_show_share=False):
    """Volume and positive share answer different questions — how many people
    fall in each category vs. what share of all positives they account for —
    so each gets its own bar chart, side by side, rather than only showing the
    share and leaving the reader to guess whether it reflects real burden or a
    small base. height defaults to a fixed 380px for horizontal charts, growing
    for breakdowns with enough categories that a fixed height would cramp the
    bars/labels (e.g. a 13-category entry-point list); pass an explicit height
    to override. By default, a category with zero positives would show up as
    an empty, uninformative bar on the share panel, so those rows are dropped
    from that panel only — the Volume panel still shows every category — and
    if that leaves fewer than two categories with any positives at all, the
    share panel is skipped entirely and Volume renders alone, full width.
    Pass always_show_share=True to opt out of that skip and always render both
    panels side by side even at 0%/100% — for a breakdown like risk-assessment
    status, the 0% bar itself (paired with the "very few positive cases" note)
    is the finding, so it should stay visible rather than being replaced."""
    chart_height = height or (max(380, 34 * len(df) + 60) if horizontal else 380)
    share_df = df if always_show_share else df[df["TOTAL_POSITIVE"] > 0]
    vol_df = df.sort_values("TOTAL_TESTED", ascending=False)
    share_df = share_df.sort_values("POSITIVITY_RATE_PCT", ascending=False)
    if not always_show_share and share_df["CATEGORY"].nunique() < 2:
        bar_chart(vol_df, "CATEGORY", "TOTAL_TESTED", f"Volume by {dimension_label}",
                   color=SEQUENTIAL_BLUE, horizontal=horizontal, height=chart_height)
        return
    vol_col, share_col = st.columns(2)
    with vol_col:
        bar_chart(vol_df, "CATEGORY", "TOTAL_TESTED", f"Volume by {dimension_label}",
                   color=SEQUENTIAL_BLUE, horizontal=horizontal, height=chart_height)
    with share_col:
        bar_chart(share_df, "CATEGORY", "POSITIVITY_RATE_PCT", f"Positive share by {dimension_label}",
                   color=color, horizontal=horizontal, pct=True, height=chart_height)


def stacked_share_bar(df, cat_col, value_col, title, colors=None, height=250):
    """A single 100%-composition bar — every category as one stacked segment on
    one row — for genuinely part-to-whole data (individual vs. couple, discordant
    vs. not). This is the form the data's job actually calls for, in place of two
    independent bars that invite an apples-to-oranges magnitude comparison.
    The shared plotly_template legend sits ABOVE the plot, in the same narrow
    top band as the title — fine for taller charts, but this one is just a
    single thin bar row, so title and legend were crowding into and overlapping
    each other. The legend is moved below the bar instead (see the layout
    override after plotly_template), which needs its own bottom margin — hence
    height defaults to 250 rather than the general chart default."""
    colors = colors or CATEGORICAL
    total = df[value_col].sum()
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        pct = 100.0 * row[value_col] / total if total else 0
        label = f"{row[cat_col]} — {pct:.0f}%"
        fig.add_bar(
            y=[title], x=[row[value_col]], name=str(row[cat_col]), orientation="h",
            marker=dict(color=colors[i % len(colors)], line=dict(color=SURFACE, width=2)),
            text=label if pct >= 8 else None, textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff"),
            hovertemplate=f"{row[cat_col]}: {row[value_col]:,.0f} ({pct:.1f}%)<extra></extra>",
        )
    fig.update_layout(title=title, height=height, barmode="stack", showlegend=True)
    fig = plotly_template(fig)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="left", x=0),
                        margin=dict(t=64, b=70, l=10, r=10))
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)
    st.plotly_chart(fig, width="stretch")


def gauge_meter(value, title, target, warn_at, higher_is_better=True, height=220, suffix="%"):
    """A single ratio measured against a limit (linkage rate vs. a 95% benchmark,
    flag rate vs. a QA threshold) — rendered as a meter, not a bare number, so the
    reader sees the remaining headroom or shortfall, not just the current value.
    Per the meter contract: the fill carries severity (good/warning/critical); the
    unfilled track stays a light, neutral step so state reads across the whole bar."""
    color = status_color(value, target, warn_at, higher_is_better=higher_is_better)
    track = "#dce8f7"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if pd.isna(value) else float(value),
        number={"suffix": suffix, "font": {"color": INK["primary"], "size": 30}},
        title={"text": title, "font": {"color": INK["secondary"], "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": INK["muted"], "tickfont": {"color": INK["muted"]}},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": track,
            "borderwidth": 0,
            "threshold": {"line": {"color": INK["primary"], "width": 2}, "thickness": 0.85, "value": target},
        },
    ))
    fig.update_layout(height=height, paper_bgcolor=SURFACE, margin=dict(l=20, r=20, t=50, b=10),
                        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif"))
    st.plotly_chart(fig, width="stretch")


def daily_pattern_chart(daily_df, avg_reference, value_col, title, y_title, color):
    """A day-by-day view of ONE month against that metric's period average — the
    visual test for whether an anomaly was a short, concentrated event (a handful of
    tall bars) or a sustained shift (every day a little higher/lower than the line)."""
    fig = go.Figure()
    fig.add_bar(x=daily_df["VISIT_DATE"], y=daily_df[value_col], marker_color=color, name=title)
    if avg_reference is not None and not pd.isna(avg_reference):
        fig.add_hline(y=avg_reference, line_dash="dash", line_color=INK["muted"],
                       annotation_text=f"Period average: {avg_reference:.1f}", annotation_position="top left")
    fig.update_layout(title=title, height=320)
    fig = plotly_template(fig, y_title=y_title)
    st.plotly_chart(fig, width="stretch")


def table_view(df, caption=None):
    st.dataframe(df, width="stretch", hide_index=True)
    if caption:
        st.caption(caption)


def note(text):
    """A short, muted, teal-railed callout for descriptive-analysis text —
    used throughout to explain *why* a chart looks the way it does, not just
    to restate what it shows."""
    st.markdown(f'<div class="insight-note">{text}</div>', unsafe_allow_html=True)


def recommend(text):
    """A navy-railed callout distinct from note() — used specifically for an
    actionable recommendation, so it reads differently from a plain observation."""
    st.markdown(f'<div class="recommend-note"><b>Recommendation:</b> {text}</div>', unsafe_allow_html=True)

# 3b. DESCRIPTIVE

def insight_categorical(df, label, cat_col="CATEGORY", rate_col="POSITIVITY_RATE_PCT",
                          n_col="TOTAL_POSITIVE", small_n=15, unit="Share of positive cases"):
    """rate_col holds each category's share of the WHOLE breakdown's total
    positives (positives in this category ÷ positives across every category),
    not that category's own hit rate — see _share_of_positive. n_col/small_n
    flag a category whose share rests on very few actual positive cases, so a
    striking-looking share isn't read as more solid than the count behind it."""
    d = df.dropna(subset=[rate_col, cat_col])
    if len(d) < 2:
        return f"Not enough categories in the selected period to compare {label}."
    top = d.loc[d[rate_col].idxmax()]
    bottom = d.loc[d[rate_col].idxmin()]
    if bottom[rate_col] > 0:
        ratio_txt = f", roughly {top[rate_col] / bottom[rate_col]:.1f}x higher"
    else:
        ratio_txt = ""
    caveats = [f"<b>{row[cat_col]}</b> rests on very few positive cases (n={int(row[n_col])})"
               for _, row in pd.DataFrame([top, bottom]).iterrows() if row[n_col] < small_n]
    caveat_txt = f" Treat with caution — {' and '.join(caveats)}." if caveats else ""
    return (f"{unit} is highest for <b>{top[cat_col]}</b> at {top[rate_col]:.1f}% and lowest for "
            f"<b>{bottom[cat_col]}</b> at {bottom[rate_col]:.1f}%{ratio_txt}.{caveat_txt}")


def insight_yield_vs_volume(df, label, cat_col="CATEGORY", rate_col="TRUE_RATE_PCT",
                              vol_col="TOTAL_TESTED", min_n=15):
    """Flags a volume/yield mismatch — the channel carrying the most testing
    isn't necessarily the one finding the most positives per test. Positive share
    alone can't surface this (a high-volume, low-yield channel still racks up a
    large share of positives from bulk testing), which is exactly why this
    reads TRUE_RATE_PCT (that channel's own hit rate) instead."""
    d = df.dropna(subset=[rate_col, cat_col, vol_col])
    if len(d) < 2:
        return None
    top_volume = d.loc[d[vol_col].idxmax()]
    top_yield = d.loc[d[rate_col].idxmax()]
    total_vol = d[vol_col].sum()
    vol_share = 100.0 * top_volume[vol_col] / total_vol if total_vol else 0
    if top_volume[cat_col] == top_yield[cat_col]:
        return (f"<b>{top_volume[cat_col]}</b> leads {label} on both fronts — the most volume "
                f"({vol_share:.0f}% of all testing here) and the highest yield ({top_volume[rate_col]:.1f}%).")
    caveat = f" (small sample, n={int(top_yield[vol_col])})" if top_yield[vol_col] < min_n else ""
    return (f"<b>{top_volume[cat_col]}</b> carries the most volume in {label} — {vol_share:.0f}% of all "
            f"testing here — but yields just {top_volume[rate_col]:.1f}% positivity, well below "
            f"<b>{top_yield[cat_col]}</b>'s {top_yield[rate_col]:.1f}%{caveat}. Most of the testing effort "
            f"here is going toward the less productive channel.")


def insight_trend(df, val_col="POSITIVITY_RATE_PCT", label="positivity rate", unit="%"):
    d = df.dropna(subset=[val_col])
    if len(d) < 4:
        return f"Not enough months in the selected period to describe a {label} trend."
    half = len(d) // 2
    first_avg, second_avg = d[val_col].iloc[:half].mean(), d[val_col].iloc[half:].mean()
    if abs(second_avg - first_avg) < max(0.05 * abs(first_avg), 0.05):
        return f"Over the selected period, {label} held roughly steady, averaging {d[val_col].mean():.1f}{unit}."
    direction = "rose" if second_avg > first_avg else "fell"
    return (f"Over the selected period, {label} {direction} from an average of {first_avg:.1f}{unit} in the "
            f"first half to {second_avg:.1f}{unit} in the second half.")


def insight_linkage(kpis, target=95.0):
    rate = kpis.LINKAGE_RATE_PCT
    if pd.isna(rate):
        return "No positive results in the selected period, so a linkage rate can't be calculated."
    unlinked = int(kpis.TOTAL_POSITIVE) - int(kpis.POSITIVES_LINKED)
    gap = target - rate
    if gap <= 0:
        return (f"Linkage to care is <b>{rate:.1f}%</b>, at or above the {target:.0f}% benchmark commonly used "
                f"in national HIV programs. {unlinked} positive client(s) are still unlinked and worth a "
                f"follow-up check regardless.")
    return (f"Linkage to care is <b>{rate:.1f}%</b>, {gap:.1f} points below the {target:.0f}% benchmark commonly "
            f"used in national HIV programs — {unlinked} positive client(s) need follow-up to close that gap.")


def insight_confirmatory(conf):
    total = int(conf.TOTAL_POSITIVE)
    if total == 0:
        return "No positive results in the selected period."
    t2, t3 = int(conf.HAS_TEST2_RESULT), int(conf.HAS_TEST3_RESULT)
    missing2 = total - t2
    txt = f"<b>{100.0 * t2 / total:.1f}%</b> of positive results ({t2} of {total}) carry a second confirmatory test."
    if missing2 > 0:
        txt += f" {missing2} positive result(s) are missing that second test and should be checked."
    return txt


def insight_algorithm_flags(alg):
    total_issues = (int(alg.INVALID_POSITIVE_ALGORITHM) + int(alg.INVALID_NEGATIVE_ALGORITHM)
                     + int(alg.DISCORDANT_CONFIRMATORY) + int(alg.EXPIRED_KIT_USED))
    if total_issues == 0:
        return "No algorithm-application or expired-kit issues were found in the selected period — a clean result."
    parts = []
    if alg.EXPIRED_KIT_USED:
        parts.append(f"{int(alg.EXPIRED_KIT_USED)} test(s) used an expired kit")
    if alg.INVALID_POSITIVE_ALGORITHM:
        parts.append(f"{int(alg.INVALID_POSITIVE_ALGORITHM)} positive result(s) didn't follow the valid algorithm")
    if alg.INVALID_NEGATIVE_ALGORITHM:
        parts.append(f"{int(alg.INVALID_NEGATIVE_ALGORITHM)} negative result(s) didn't follow the valid algorithm")
    if alg.DISCORDANT_CONFIRMATORY:
        parts.append(f"{int(alg.DISCORDANT_CONFIRMATORY)} confirmatory result(s) were discordant")
    return "In the selected period: " + "; ".join(parts) + "."


def insight_leading_flag(flag_df, total_encounters):
    top = flag_df.iloc[0]
    pct = 100.0 * top["COUNT"] / total_encounters if total_encounters else 0
    return (f"The most common data-quality issue is <b>{top['CATEGORY']}</b>, affecting {int(top['COUNT']):,} "
            f"encounters ({pct:.1f}% of all encounters in the period) — the first place to focus a QA review.")


def insight_couple(couple_df, discordance_df):
    total = couple_df["TOTAL_ENCOUNTERS"].sum()
    couple_row = couple_df[couple_df["CATEGORY"] == "Couple"]
    couple_n = int(couple_row["TOTAL_ENCOUNTERS"].iloc[0]) if not couple_row.empty else 0
    share = 100.0 * couple_n / total if total else 0
    disc_mask = discordance_df["CATEGORY"].isin([True, "Yes", "yes", "TRUE", "true"])
    disc_row = discordance_df[disc_mask]
    disc_n = int(disc_row["TOTAL_COUPLES"].sum()) if not disc_row.empty else 0
    disc_total = discordance_df["TOTAL_COUPLES"].sum()
    disc_pct = 100.0 * disc_n / disc_total if disc_total else 0
    return (f"Couples make up <b>{share:.1f}%</b> of all encounters ({couple_n:,} of {int(total):,}). Among couples "
            f"tested together, <b>{disc_pct:.1f}%</b> ({disc_n} of {int(disc_total)}) were discordant — one partner "
            f"positive, one negative — the group most in need of targeted counseling.")


def insight_provider_outliers(prov_df, min_n=30):
    d = prov_df[prov_df["TOTAL_TESTED"] >= min_n]
    if len(d) < 2:
        return "Not enough providers with sufficient volume in this period to compare performance."
    top_flag = d.loc[d["FLAG_RATE_PCT"].idxmax()]
    low_link = d.loc[d["LINKAGE_RATE_PCT"].idxmin()]
    median_flag = d["FLAG_RATE_PCT"].median()
    return (f"Provider <b>{top_flag['PROVIDER_ID'][:10]}…</b> has the highest flag rate at "
            f"{top_flag['FLAG_RATE_PCT']:.1f}%, against a {median_flag:.1f}% median across {len(d)} providers "
            f"with at least {min_n} tests — worth a data-quality check. Provider "
            f"<b>{low_link['PROVIDER_ID'][:10]}…</b> has the lowest linkage rate at {low_link['LINKAGE_RATE_PCT']:.1f}%.")


def insight_unlinked_by_provider(unlinked):
    """Names the provider carrying the most unlinked positive clients — the
    finding the per-provider chart makes visible before the raw list backs it up."""
    by_provider = unlinked.groupby("PROVIDER_ID").size().sort_values(ascending=False)
    top_provider, top_count = by_provider.index[0], int(by_provider.iloc[0])
    return (f"Provider <b>{top_provider[:10]}…</b> has the most unlinked positive clients — "
            f"<b>{top_count}</b> of {len(unlinked)} total unlinked cases in this period "
            f"({100.0 * top_count / len(unlinked):.0f}%) — the first place to direct follow-up capacity.")


def render_unlinked_detail(unlinked, chart_title="Unlinked positive clients by provider"):
    """The visual -> finding -> detail flow for the unlinked-positives list, reused
    everywhere the dashboard surfaces it: a chart of the count per provider first
    (so the pattern is visible), then the note naming the standout, then the raw
    per-client rows for follow-up — never the raw list on its own."""
    if unlinked.empty:
        note("No unlinked positive clients in the selected period — every positive result was linked to care.")
        return
    by_provider = (unlinked.groupby("PROVIDER_ID").size().sort_values(ascending=False)
                    .reset_index(name="UNLINKED_COUNT"))
    by_provider_display = by_provider.copy()
    by_provider_display["PROVIDER_ID"] = by_provider_display["PROVIDER_ID"].str[:10] + "…"
    bar_chart(by_provider_display, "PROVIDER_ID", "UNLINKED_COUNT", chart_title, color=CATEGORICAL[7],
               horizontal=True, height=min(500, max(220, 50 * len(by_provider_display))))
    note(insight_unlinked_by_provider(unlinked))
    detail = unlinked.sort_values("PROVIDER_ID").copy()
    detail["PROVIDER_ID"] = detail["PROVIDER_ID"].str[:10] + "…"
    table_view(
        detail,
        caption=f"All {len(unlinked)} unlinked positive clients in the selected period, grouped by "
                 "provider — use this list to drive outreach and close the linkage gap.",
    )


def detect_spikes(df, value_col, month_col="VISIT_MONTH", z_thresh=1.3, max_n=2):
    """Flag months where value_col is an outlier against the period's own mean/std
    (a z-score check) — used to find volume/positivity spikes worth explaining,
    rather than reacting to every up-and-down wiggle in the line."""
    d = df.dropna(subset=[value_col]).copy()
    if len(d) < 4:
        return []
    std = d[value_col].std()
    if not std or pd.isna(std):
        return []
    d["Z"] = (d[value_col] - d[value_col].mean()) / std
    spikes = d[d["Z"].abs() >= z_thresh].copy()
    if spikes.empty:
        return []
    spikes = spikes.reindex(spikes["Z"].abs().sort_values(ascending=False).index)
    return spikes.head(max_n).to_dict("records")


def split_high_low_months(trend, rate_col="POSITIVITY_RATE_PCT", month_col="VISIT_MONTH", n=3, min_months=4):
    """Splits the monthly trend into the n months with the highest share of
    positive cases and the n months with the lowest — a structural comparison
    ("what's systematically different about months that run hot vs. months
    that run cool") rather than an event-level look at any one anomalous
    month or day. Returns (None, None) when there aren't enough distinct
    months for the two groups to be meaningful and non-overlapping."""
    d = trend.dropna(subset=[rate_col, month_col])
    if len(d) < min_months:
        return None, None
    n = max(1, min(n, len(d) // 2))
    ranked = d.sort_values(rate_col, ascending=False)
    high = ranked.head(n)[month_col].tolist()
    low = ranked.tail(n)[month_col].tolist()
    return high, low


def compare_high_low_mix(mix_df, high_months, low_months, cat_col="CATEGORY", month_col="VISIT_MONTH",
                            count_col="TOTAL_TESTED"):
    """Combines the high-positivity months into one group and the low-positivity
    months into another, then expresses each category's share WITHIN its own
    group's testing volume — so the two groups compare on the same 0-100%
    scale regardless of how much total testing happened in each. Long-form
    output (one row per category x group) feeds grouped_bar_chart directly."""
    empty = pd.DataFrame(columns=[cat_col, "GROUP", "SHARE_PCT"])
    if mix_df.empty or high_months is None or low_months is None:
        return empty
    d = mix_df.copy()
    d[month_col] = pd.to_datetime(d[month_col])
    high_set = {pd.Timestamp(m) for m in high_months}
    low_set = {pd.Timestamp(m) for m in low_months}
    rows = []
    for label, month_set in [("High-positivity months", high_set), ("Low-positivity months", low_set)]:
        slice_df = d[d[month_col].isin(month_set)]
        total = slice_df[count_col].sum()
        if total == 0:
            continue
        for cat, cnt in slice_df.groupby(cat_col)[count_col].sum().items():
            rows.append({cat_col: cat, "GROUP": label, "SHARE_PCT": 100.0 * cnt / total})
    return pd.DataFrame(rows) if rows else empty


def insight_high_low_comparison(comp, label, cat_col="CATEGORY", min_delta_pts=8):
    """Names the category with the biggest gap in mix share between the
    high-positivity and low-positivity month groups — the standout difference
    in HOW testing was done between months that ran hot and months that ran
    cool, the day-level equivalent of insight_composition_chart but for a
    structural month-group comparison instead of one anomalous month."""
    if comp.empty:
        return f"Not enough data in both groups to compare {label} mix."
    pivot = comp.pivot_table(index=cat_col, columns="GROUP", values="SHARE_PCT", fill_value=0)
    if "High-positivity months" not in pivot.columns or "Low-positivity months" not in pivot.columns:
        return f"Not enough data in both groups to compare {label} mix."
    pivot["DELTA"] = pivot["High-positivity months"] - pivot["Low-positivity months"]
    top_cat = pivot["DELTA"].abs().idxmax()
    row = pivot.loc[top_cat]
    if abs(row["DELTA"]) < min_delta_pts:
        return (f"No {label} stands out between the two groups — the biggest gap was <b>{top_cat}</b> at just "
                f"{abs(row['DELTA']):.0f} points, which doesn't point to {label} mix as a likely explanation.")
    direction = "more" if row["DELTA"] > 0 else "less"
    return (f"<b>{top_cat}</b> is the biggest {label} difference between the two groups — "
            f"{row['High-positivity months']:.0f}% of testing volume in high-positivity months vs. "
            f"{row['Low-positivity months']:.0f}% in low-positivity months, {abs(row['DELTA']):.0f} points "
            f"{direction} common when positivity runs high.")


def top_category_in_month(mix_df, month, cat_col="CATEGORY", month_col="VISIT_MONTH", count_col="TOTAL_TESTED"):
    """The category used most within one month, plus its share of that
    month's total — e.g. 'PITC (54%)' — used to summarize a month's
    testing-strategy or entry-point mix down to one at-a-glance detail for a
    table row, rather than a full per-category breakdown. mix_df's month
    column is re-parsed with pd.to_datetime before comparing — Snowflake can
    hand back a DATE column as plain datetime.date objects, which compare as
    NOT EQUAL to a pd.Timestamp of the same day even though they represent
    the same month, so comparing the raw column would silently match nothing."""
    d = mix_df[pd.to_datetime(mix_df[month_col]) == pd.Timestamp(month)]
    total = d[count_col].sum()
    if d.empty or total == 0:
        return "—"
    top = d.loc[d[count_col].idxmax()]
    pct = 100.0 * top[count_col] / total
    return f"{top[cat_col]} ({pct:.0f}%)"


def build_month_detail_table(months, trend, strategy_mix, entrypoint_mix, month_col="VISIT_MONTH",
                               rate_col="POSITIVITY_RATE_PCT"):
    """One row per month — its positivity share, testing volume, and the
    leading testing strategy and entry point that month — the 'corresponding
    details' for a group of high- or low-positivity months, laid out as a
    table so each month's detail is readable on its own row rather than
    folded into an aggregated bar chart."""
    t = trend.copy()
    t[month_col] = pd.to_datetime(t[month_col])
    rows = []
    for m in months:
        m_ts = pd.Timestamp(m)
        match = t[t[month_col] == m_ts]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append({
            "_rate": r[rate_col],
            "Month": m_ts.strftime("%b %Y"),
            "Positivity share": f"{r[rate_col]:.1f}%",
            "Total tested": f"{int(r['TOTAL_TESTED']):,}",
            "Testing strategy used": top_category_in_month(strategy_mix, m_ts),
            "Entry point used": top_category_in_month(entrypoint_mix, m_ts),
        })
    
    rows.sort(key=lambda row: row["_rate"], reverse=True)
    for row in rows:
        del row["_rate"]
    return pd.DataFrame(rows)


def grouped_bar_chart(df, cat_col, group_col, value_col, title, colors, horizontal=True, height=None,
                        x_title=None):
    """Two series side by side per category — for comparing the SAME set of
    categories across two different slices of data (e.g. high- vs.
    low-positivity months), where two separate single-series charts would
    force the reader to hold one chart's scale in their head while reading
    the other. Categories are ranked by the first group's value, descending,
    matching the same largest-first convention as every other chart here."""
    if df.empty:
        return
    groups = list(dict.fromkeys(df[group_col]))  # preserve first-seen order
    order_group = groups[0]
    order = (df[df[group_col] == order_group].sort_values(value_col, ascending=False)[cat_col].tolist())
    remaining = [c for c in df[cat_col].unique() if c not in order]
    categories = order + remaining
    chart_height = height or (max(380, 32 * len(categories) + 100) if horizontal else 420)
    fig = go.Figure()
    for i, g in enumerate(groups):
        gdf = df[df[group_col] == g].set_index(cat_col).reindex(categories).reset_index()
        if horizontal:
            fig.add_bar(y=gdf[cat_col], x=gdf[value_col], name=str(g), orientation="h",
                         marker_color=colors[i % len(colors)])
        else:
            fig.add_bar(x=gdf[cat_col], y=gdf[value_col], name=str(g), marker_color=colors[i % len(colors)])
    fig.update_layout(title=title, height=chart_height, barmode="group")
    fig = plotly_template(fig, y_title="%" if not horizontal else None)
    if horizontal:
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title=x_title or "% of that group's testing volume")
    st.plotly_chart(fig, width="stretch")


def composition_delta_table(mix_df, slice_value, slice_col, share_col, cat_col="CATEGORY", min_overall_count=0):
    """Every category's share of ONE slice vs. its share of the whole period,
    ranked by delta, descending — the full ranking behind explain_group_composition's
    top pick. Exposed separately so a chart can show every category's delta (even
    the ones below threshold), not just whichever one wins."""
    empty = pd.DataFrame(columns=[cat_col, "SLICE_SHARE", "OVERALL_SHARE", "DELTA_PTS"])
    if mix_df.empty or cat_col not in mix_df.columns or slice_col not in mix_df.columns:
        return empty
    overall_counts = mix_df.groupby(cat_col)[share_col].sum()
    valid_cats = overall_counts[overall_counts >= min_overall_count].index
    slice_df = mix_df[(mix_df[slice_col] == slice_value) & (mix_df[cat_col].isin(valid_cats))]
    if slice_df.empty or slice_df[share_col].sum() == 0:
        return empty
    slice_share = slice_df.groupby(cat_col)[share_col].sum()
    slice_share = slice_share / slice_share.sum()
    overall_share = overall_counts.reindex(valid_cats)
    overall_share = overall_share / overall_share.sum()
    delta = (slice_share - overall_share.reindex(slice_share.index).fillna(0)).sort_values(ascending=False)
    if delta.empty:
        return empty
    return pd.DataFrame({
        cat_col: delta.index,
        "SLICE_SHARE": (slice_share.reindex(delta.index) * 100).values,
        "OVERALL_SHARE": (overall_share.reindex(delta.index).fillna(0) * 100).values,
        "DELTA_PTS": (delta * 100).values,
    })


def explain_group_composition(mix_df, slice_value, slice_col, share_col, cat_col="CATEGORY",
                                min_delta_pts=5, min_overall_count=0):
    """Compares each category's share of ONE slice (a month, a demographic group,
    whatever slice_col picks out) against its share of the whole selected period,
    and returns whichever category is most over-represented in that slice — the
    composition difference most likely contributing to the slice standing out.
    min_overall_count drops thin categories (e.g. a low-volume provider) whose
    share can swing wildly on noise alone. This is the shared engine behind both
    the spike-cause analysis (slice = a month) and the leading-category-cause
    analysis (slice = a demographic group)."""
    table = composition_delta_table(mix_df, slice_value, slice_col, share_col, cat_col, min_overall_count)
    if table.empty or table["DELTA_PTS"].iloc[0] < min_delta_pts:
        return None
    top = table.iloc[0]
    return {
        "category": top[cat_col],
        "slice_share": top["SLICE_SHARE"],
        "overall_share": top["OVERALL_SHARE"],
        "delta_pts": top["DELTA_PTS"],
    }


def explain_month_composition(mix_df, month, share_col="TOTAL_TESTED", month_col="VISIT_MONTH",
                                cat_col="CATEGORY", min_delta_pts=5, min_overall_count=0):
    """Thin wrapper over explain_group_composition for the month-spike case —
    kept so every existing spike-analysis call site (analyze_spike_factors,
    recommend_for_spike, etc.) keeps working against the same 'month_share' key
    it already expects."""
    result = explain_group_composition(mix_df, pd.Timestamp(month), month_col, share_col, cat_col,
                                         min_delta_pts, min_overall_count)
    if result is None:
        return None
    return {"category": result["category"], "month_share": result["slice_share"],
            "overall_share": result["overall_share"], "delta_pts": result["delta_pts"]}


def analyze_spike_factors(month, share_col, dimension_mixes, min_delta_pts=5, top_n=3, require_threshold=True):
    """Runs explain_month_composition across EVERY tracked dimension (entry point,
    testing strategy, setting, age band, sex, kit brand, provider, retest status) for
    one spike month, then ranks every (dimension, category) pair by how far it swung
    from its usual share. dimension_mixes is a list of (label, mix_df, min_overall_count)
    tuples. With require_threshold=False, returns the full ranked list regardless of
    magnitude — used to chart the comparison so a flat result is visibly proven, not
    just asserted in text."""
    candidates = []
    for label, mix_df, min_overall_count in dimension_mixes:
        driver = explain_month_composition(mix_df, month, share_col=share_col, min_delta_pts=0,
                                             min_overall_count=min_overall_count)
        if driver:
            candidates.append({**driver, "dimension": label})
    if require_threshold:
        candidates = [c for c in candidates if c["delta_pts"] >= min_delta_pts]
    candidates.sort(key=lambda c: c["delta_pts"], reverse=True)
    return candidates[:top_n]


def analyze_category_factors(slice_value, factor_mixes, min_delta_pts=8, top_n=1, min_overall_count=15,
                               require_threshold=True):
    """Same engine as analyze_spike_factors, but the 'slice' being explained is a
    demographic/clinical category instead of a month — e.g. checking whether the
    age band with the highest positivity rate is reached disproportionately
    through a particular entry point or testing strategy, which would mean
    channel mix is doing some of the work rather than age itself. Compares each
    factor category's share of the SLICE'S POSITIVE CASES (TOTAL_POSITIVE)
    against its share of positive cases overall — not its share of testing
    volume — since the thing being explained is why a group's rate of
    positivity stands out, not why its testing volume does. factor_mixes is a
    list of (label, two_dim_mix_df) tuples, each keyed by SLICE_CATEGORY. With
    require_threshold=False, returns the full ranked list regardless of
    magnitude — used to chart the comparison so a flat result is visibly proven,
    not just asserted in text."""
    candidates = []
    for label, mix_df in factor_mixes:
        driver = explain_group_composition(mix_df, slice_value, "SLICE_CATEGORY", "TOTAL_POSITIVE",
                                             cat_col="CATEGORY", min_delta_pts=0,
                                             min_overall_count=min_overall_count)
        if driver:
            candidates.append({**driver, "dimension": label})
    if require_threshold:
        candidates = [c for c in candidates if c["delta_pts"] >= min_delta_pts]
    candidates.sort(key=lambda c: c["delta_pts"], reverse=True)
    return candidates[:top_n]


def render_category_driver(df, factor_mixes, cat_col="CATEGORY", rate_col="POSITIVITY_RATE_PCT", threshold=8,
                             always_show=False):
    """Diagnostic follow-up to insight_categorical: rather than just naming which
    category leads, charts EVERY tracked channel's (entry point, testing
    strategy, setting, age band) share of that leading category's POSITIVE
    CASES against its usual share of positive cases, ranked — the same 'is this
    a real effect or a confound' visual the spike deep-dive already draws for a
    month, so 'nothing stands out' is something the audience can see, not just
    be told. rate_col defaults to POSITIVITY_RATE_PCT (positive share) — the SAME
    metric shown in the Positive-share chart directly above this one — so the
    category this diagnostic explains is always the one the audience can
    already see leading, never a different category picked out by a metric
    (like the true per-capita rate) that isn't otherwise displayed here.
    By default, renders nothing if there's no category to compare (fewer than
    two rows), no factor data at all, or if no tracked factor actually crosses
    the threshold (a chart where every bar sits near zero has nothing to show).
    Pass always_show=True to keep this diagnostic on screen even when nothing
    crosses the threshold — for a dimension like sex, the audience expects to
    always see this deep-dive rather than have it silently disappear some
    periods; the "nothing stands out" finding is then stated in the note
    instead of hidden."""
    d = df.dropna(subset=[rate_col, cat_col])
    if len(d) < 2:
        return
    top = d.loc[d[rate_col].idxmax()]
    drivers = analyze_category_factors(top[cat_col], factor_mixes, require_threshold=False, top_n=8)
    if not drivers:
        return
    driver = drivers[0]
    if driver["delta_pts"] < threshold and not always_show:
        return
    comp_df = pd.DataFrame([{"CATEGORY": f"{dr['dimension']}: {dr['category']}", "DELTA_PTS": dr["delta_pts"]}
                              for dr in drivers])
    bar_chart(comp_df, "CATEGORY", "DELTA_PTS",
               f"Why {top[cat_col]} leads on positive share — positive-case mix vs. usual share",
               color=CATEGORICAL[7], horizontal=True, height=280,
               x_title="Percentage points above usual share of positive cases")
    if driver["delta_pts"] >= threshold:
        note(f"Part of why <b>{top[cat_col]}</b> leads on positive share may be channel mix, not the group itself: "
             f"<b>{driver['slice_share']:.0f}%</b> of its positive cases come via "
             f"<b>{driver['category']}</b> ({driver['dimension']}), vs {driver['overall_share']:.0f}% of positive "
             f"cases typically — a {driver['delta_pts']:.0f}-point gap worth ruling out before treating "
             f"{top[cat_col]} itself as the driver.")
    else:
        note(f"Nothing crosses a meaningful threshold here — the largest positive-case mix shift behind "
             f"<b>{top[cat_col]}</b> was <b>{driver['category']}</b> ({driver['dimension']}) at just "
             f"{driver['delta_pts']:.0f} points above its usual share of positive cases, which rules out channel "
             f"mix as a likely driver.")


def quality_caveat_for_month(month, flag_trend, z_thresh=1.3):
    """True if the data-quality flag rate was itself a statistical spike that same
    month — a signal that part of the anomaly may be a data-entry artifact rather
    than a genuine programmatic or clinical shift."""
    flag_spikes = detect_spikes(flag_trend, "FLAG_RATE_PCT", z_thresh=z_thresh, max_n=len(flag_trend))
    spike_months = {pd.Timestamp(s["VISIT_MONTH"]) for s in flag_spikes}
    return pd.Timestamp(month) in spike_months


def insight_daily_pattern(daily_df, count_col, share_of):
    """Was the change concentrated in a handful of days (a discrete event: outreach,
    a mobile clinic, a data-loading batch) or spread across the whole month (a
    sustained shift: staffing, a new provider, a policy or reporting change)? This is
    the difference between 'go look at what happened on these dates' and 'go look for
    something that changed structurally that month'."""
    d = daily_df.dropna(subset=[count_col])
    if d.empty or d[count_col].sum() == 0:
        return ("No daily records were found for this month to check whether it was a short event or a "
                 "sustained shift."), False, "", []
    k = min(3, len(d))
    top = d.nlargest(k, count_col)
    top_share = 100.0 * top[count_col].sum() / d[count_col].sum()
    top_date_values = pd.to_datetime(top["VISIT_DATE"]).dt.date.tolist()
    top_dates = ", ".join(pd.to_datetime(top["VISIT_DATE"]).dt.strftime("%d %b").tolist())
    concentrated = top_share >= 40
    if concentrated:
        text = (f"Just {k} day(s) — <b>{top_dates}</b> — account for <b>{top_share:.0f}%</b> of the month's "
                f"{share_of}, pointing to a short, concentrated event rather than a sustained shift across "
                f"the month.")
    else:
        text = (f"The {k} busiest days only account for {top_share:.0f}% of the month's {share_of} — the "
                f"change is spread fairly evenly across the month, more consistent with a sustained shift "
                f"than a single event.")
    return text, concentrated, top_dates, top_date_values


def insight_daily_strategy(strategy_df, top_dates):
    """Names the testing strategy that actually carried a spike's concentrated
    day(s) — the level below insight_composition_chart's month-wide comparison.
    Only meaningful once the spike has already been shown to be concentrated on
    specific dates (see `concentrated` in insight_daily_pattern); returns None
    when there's no positive-result data for those dates to rank by."""
    d = strategy_df.dropna(subset=["TOTAL_POSITIVE"])
    total_positive = d["TOTAL_POSITIVE"].sum()
    if d.empty or total_positive == 0:
        return None
    top = d.loc[d["TOTAL_POSITIVE"].idxmax()]
    total_tested = d["TOTAL_TESTED"].sum()
    pos_share = 100.0 * top["TOTAL_POSITIVE"] / total_positive
    vol_share = 100.0 * top["TOTAL_TESTED"] / total_tested if total_tested else 0.0
    return (f"On {top_dates}, <b>{top['CATEGORY']}</b> was the leading testing strategy behind the spike — "
            f"it accounted for <b>{int(top['TOTAL_POSITIVE'])} of the {int(total_positive)}</b> positive "
            f"results on those day(s) ({pos_share:.0f}%), from {vol_share:.0f}% of that day's testing volume.")


def insight_composition_chart(drivers, share_of, threshold=5):
    """Reads the ranked, unfiltered factor list — names the top mover if it's large
    enough to matter, or explicitly rules out every tracked factor if not, so the
    chart's flatness is stated as a finding rather than left to interpretation."""
    if not drivers:
        return "No category breakdown was available to compare for this month."
    top = drivers[0]
    if top["delta_pts"] >= threshold:
        return (f"<b>{top['category']}</b> ({top['dimension']}) stands out — {top['month_share']:.0f}% of "
                f"that month's {share_of} vs. {top['overall_share']:.0f}% typically, a "
                f"{top['delta_pts']:.0f}-point jump large enough to treat as a likely contributor.")
    return (f"Nothing crosses a meaningful threshold here — the largest shift was <b>{top['category']}</b> "
            f"({top['dimension']}) at just {top['delta_pts']:.0f} points above its usual share, which rules "
            f"out any single demographic group, provider, or testing channel as the driver.")


DIMENSION_RECOMMENDATIONS = {
    "Entry point": "confirm whether a specific outreach or campaign ran at this entry point in {month}; if it "
                    "drove real reach, consider repeating or scaling it",
    "Testing strategy": "check whether this testing strategy was deliberately scaled up in {month} — if it's "
                         "proving effective, consider shifting more resources toward it",
    "Provider": "follow up directly with this provider on whether {month} reflects a genuine outreach event or "
                "a data-loading/backfill batch, and replicate the approach elsewhere if genuine",
    "Setting": "verify whether a community outreach event took place in {month} — if so, it's a candidate to "
               "repeat in future quarters",
    "Age band": "coordinate with programs targeting this age group to understand and sustain what drove the "
                "{month} increase",
    "Sex": "coordinate with programs targeting this group specifically to understand what drove the {month} "
           "increase",
    "Kit brand": "run a QC check on this kit brand's lot number and expiry for {month} before treating the "
                 "shift as a true epidemiological signal",
    "Retest status": "this looks like a wave of first-time testers — check the program calendar for an "
                      "awareness or outreach push around {month}",
}


def recommend_for_spike(month, drivers, quality_flag, concentrated, top_dates, threshold=5, daily_strategy=None):
    """Synthesizes all three angles — daily concentration, factor composition, and
    data-quality context — into one actionable recommendation, so it never rests on
    just one lens. daily_strategy (when the spike is concentrated on specific days)
    names the testing strategy that carried those days, so the recommendation can
    point at that strategy specifically rather than just "check the calendar"."""
    month_str = f"{pd.Timestamp(month):%B %Y}"
    lines = []
    if quality_flag:
        lines.append(f"the data-quality flag rate also jumped in {month_str} — rule out duplicate or "
                       "backfilled records for that month before treating this as a genuine trend")
    if concentrated and top_dates:
        if daily_strategy:
            lines.append(f"the change is concentrated on {top_dates}, driven mainly by <b>{daily_strategy}</b> — "
                           f"check whether that strategy was deliberately scaled up (an outreach push or mobile "
                           f"clinic event) on those specific dates, and if so, consider repeating or scaling it")
        else:
            lines.append(f"the change is concentrated on {top_dates} — check the outreach/event calendar for "
                           "those specific dates, and if it was a planned activity, consider repeating or scaling it")
    else:
        lines.append(f"the change is spread across {month_str} rather than a few days — look for a sustained "
                       "cause over that month (a new provider coming online, a staffing change, or a shift in "
                       "reporting or registration practice) rather than a single event")
    top = drivers[0] if drivers else None
    if top and top["delta_pts"] >= threshold:
        template = DIMENSION_RECOMMENDATIONS.get(top["dimension"])
        if template:
            lines.append(template.format(month=month_str))
        else:
            lines.append(f"look into {top['category']} within {top['dimension']} for {month_str} to confirm "
                           "what changed")
    else:
        lines.append("no demographic, provider, entry-point, testing-strategy, or kit-brand factor stands "
                       "out either, reinforcing that this looks structural rather than tied to one group or "
                       "channel")
    combined = "; and ".join(lines)
    return combined[0].upper() + combined[1:] + "."

# 4. SQL QUERY FUNCTIONS  

def _date_cond(start, end):
    if start and end:
        return f"VISIT_DATE_D BETWEEN '{start}' AND '{end}'"
    return None


def _where(*conditions):
    conds = [c for c in conditions if c]
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def _fillna_category(df, *cols, label="Not recorded"):
    """Snowflake returns NULL for an unset categorical field, which pandas loads
    as NaN — left alone, that produces an unlabeled bar in the chart and a
    literal 'nan' in the generated insight text (the chart and the footnote
    disagreeing about what's on screen). Replacing it with a real label keeps
    the two in sync and keeps the group visible instead of silently vanishing."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(label)
    return df


def _share_of_positive(df, total_col="TOTAL_POSITIVE", out_col="POSITIVITY_RATE_PCT"):
    """Overwrites out_col with each row's share of the WHOLE result set's total
    positives (rows sum to ~100%), rather than that row's own hit rate
    (its positives ÷ its own tests). Every caller's rows already partition the
    same filtered population by one dimension (month, provider, category, ...),
    so this reads as 'what share of all positive cases came from here' — a
    composition/burden view, not a per-capita risk rate. Applied uniformly so a
    chart and its footnote never disagree on which of the two this number is."""
    total = df[total_col].sum() if total_col in df.columns else 0
    df[out_col] = 100.0 * df[total_col] / total if total else 0.0
    return df


def _add_true_rate(df, positive_col="TOTAL_POSITIVE", total_col="TOTAL_TESTED", out_col="TRUE_RATE_PCT"):
    """Adds each row's own hit rate (its positives ÷ its own tests) — the
    per-capita risk measure that POSITIVITY_RATE_PCT stopped being once it was
    redefined as a share of the whole breakdown's positives. Kept as a
    separate column so a chart can show either, or both, without the two ever
    being confused for one another."""
    df[out_col] = (100.0 * df[positive_col] / df[total_col]).where(df[total_col] > 0, 0.0)
    return df


def get_date_bounds():
    sql = f"SELECT MIN(VISIT_DATE_D) AS MIN_DATE, MAX(VISIT_DATE_D) AS MAX_DATE FROM {TABLE};"
    return run_query(sql, "slow")


def get_kpis(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            COUNT(*)                                                    AS total_tested,
            COUNT(DISTINCT PATIENT_ID)                                  AS unique_clients,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                             AS positivity_rate_pct,
            SUM(CASE WHEN FINALRESULT = 'Positive' AND IS_LINKED = TRUE THEN 1 ELSE 0 END) AS positives_linked,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' AND IS_LINKED = TRUE THEN 1 ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END), 0), 2) AS linkage_rate_pct,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULTGIVEN = TRUE THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                             AS result_given_rate_pct,
            MIN(VISIT_DATE_D)                                           AS earliest_visit,
            MAX(VISIT_DATE_D)                                           AS latest_visit
        FROM {TABLE}
        {where_sql};
    """
    return run_query(sql, "fast")


def get_monthly_trend(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            DATE_TRUNC('MONTH', VISIT_DATE_D)                           AS visit_month,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive
        FROM {TABLE}
        {where_sql}
        GROUP BY DATE_TRUNC('MONTH', VISIT_DATE_D)
        ORDER BY 1;
    """
    return _share_of_positive(run_query(sql, "fast"))


def get_monthly_dimension_mix(column, start=None, end=None):
    """Volume + positives by month AND by category of `column` — used to explain a
    spike in the monthly trend by seeing which category's share of that month's
    activity was unusually high compared to its share across the whole period."""
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            DATE_TRUNC('MONTH', VISIT_DATE_D)                           AS visit_month,
            {column}                                                    AS category,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive
        FROM {TABLE}
        {where_sql}
        GROUP BY DATE_TRUNC('MONTH', VISIT_DATE_D), {column}
        ORDER BY 1, 2;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY")


def get_two_dim_mix(slice_column, factor_column, start=None, end=None):
    """Cross-tab of slice_column x factor_column — lets a demographic or clinical
    group's distinctively high true rate be checked against its distribution
    across another tracked channel (entry point, testing strategy, setting, age
    band), the same way get_monthly_dimension_mix lets a month's spike be
    checked against its distribution across a channel. Used by
    render_category_driver to test whether a group's rate is (partly) a
    channel-mix confound rather than a real effect of the group itself."""
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            {slice_column}  AS slice_category,
            {factor_column} AS category,
            COUNT(*)        AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END) AS total_positive
        FROM {TABLE}
        {where_sql}
        GROUP BY {slice_column}, {factor_column}
        ORDER BY 1, 2;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY", "SLICE_CATEGORY")


_DRIVER_FACTOR_COLS = [
    ("Entry point", "TESTENTRYPOINT_LABEL"),
    ("Testing strategy", "TESTINGSTRATEGY_LABEL"),
    ("Setting", "SETTING"),
    ("Age band", "AGE_BAND"),
]


def get_driver_factor_mixes(dimension_col, start=None, end=None):
    """The two-dim mixes render_category_driver checks a dimension's leading
    category against — every entry in _DRIVER_FACTOR_COLS except the dimension
    being explained itself (comparing a dimension against itself is circular)."""
    return [
        (label, get_two_dim_mix(dimension_col, col, start, end))
        for label, col in _DRIVER_FACTOR_COLS
        if col != dimension_col
    ]


def get_monthly_entrypoint_mix(start=None, end=None):
    return get_monthly_dimension_mix("TESTENTRYPOINT_LABEL", start=start, end=end)


def get_monthly_strategy_mix(start=None, end=None):
    return get_monthly_dimension_mix("TESTINGSTRATEGY_LABEL", start=start, end=end)


def get_monthly_retest_mix(start=None, end=None):
    """Same shape as get_monthly_dimension_mix, but bucketed by time-since-last-test —
    lets a spike explanation check whether a month skewed toward first-time testers."""
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            DATE_TRUNC('MONTH', VISIT_DATE_D) AS visit_month,
            CASE
                WHEN MONTHSSINCELASTTEST_NUM IS NULL THEN 'No prior test / not recorded'
                WHEN MONTHSSINCELASTTEST_NUM <= 3  THEN '0-3 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 6  THEN '4-6 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 12 THEN '7-12 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 24 THEN '13-24 months'
                ELSE '24+ months'
            END                                                          AS category,
            COUNT(*)                                                     AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)    AS total_positive
        FROM {TABLE}
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 1, 2;
    """
    return run_query(sql, "fast")


def get_daily_breakdown_for_month(month):
    """Every day's volume + positivity within ONE calendar month — the deep-dive
    query behind the spike investigation: was the anomaly a few unusually busy/quiet
    days, or a level shift that held across the whole month? Scoped to the month
    itself rather than the sidebar's date range, since the month was already chosen
    from within that range."""
    month_ts = pd.Timestamp(month)
    month_start = month_ts.date()
    month_end = (month_ts + pd.DateOffset(months=1)).date()
    sql = f"""
        SELECT
            VISIT_DATE_D                                                AS visit_date,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                             AS positivity_rate_pct
        FROM {TABLE}
        WHERE VISIT_DATE_D >= '{month_start}' AND VISIT_DATE_D < '{month_end}'
        GROUP BY VISIT_DATE_D
        ORDER BY 1;
    """
    return run_query(sql, "fast")


def get_strategy_mix_for_dates(dates):
    """Testing-strategy breakdown (volume + positives) for one or more specific
    calendar dates — the day-level counterpart to get_monthly_strategy_mix, used
    once a spike has been traced to a handful of concentrated days, to name which
    strategy was actually being run on those days rather than stopping at "the
    month's testing strategy mix looked like X"."""
    if not dates:
        return pd.DataFrame(columns=["CATEGORY", "TOTAL_TESTED", "TOTAL_POSITIVE"])
    date_list = ", ".join(f"'{pd.Timestamp(d).date()}'" for d in dates)
    sql = f"""
        SELECT
            TESTINGSTRATEGY_LABEL                                       AS category,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive
        FROM {TABLE}
        WHERE VISIT_DATE_D IN ({date_list})
        GROUP BY TESTINGSTRATEGY_LABEL
        ORDER BY total_positive DESC;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY")


def get_positivity_by(column, order_by="total_tested", start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            {column}                                                    AS category,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive
        FROM {TABLE}
        {where_sql}
        GROUP BY {column}
        ORDER BY {order_by.replace('positivity_rate_pct', 'total_positive')} DESC;
    """
    
    df = _add_true_rate(_share_of_positive(_fillna_category(run_query(sql, "fast"), "CATEGORY")))
    return df


def get_strategy_yield(start=None, end=None):
    return get_positivity_by("TESTINGSTRATEGY_LABEL", order_by="positivity_rate_pct", start=start, end=end)


def get_entrypoint_yield(start=None, end=None):
    return get_positivity_by("TESTENTRYPOINT_LABEL", order_by="positivity_rate_pct", start=start, end=end)


def get_retest_interval(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            CASE
                WHEN MONTHSSINCELASTTEST_NUM IS NULL THEN 'No prior test / not recorded'
                WHEN MONTHSSINCELASTTEST_NUM <= 3  THEN '0-3 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 6  THEN '4-6 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 12 THEN '7-12 months'
                WHEN MONTHSSINCELASTTEST_NUM <= 24 THEN '13-24 months'
                ELSE '24+ months'
            END                                                          AS category,
            COUNT(*)                                                     AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)    AS total_positive,
            MIN(COALESCE(MONTHSSINCELASTTEST_NUM, -1))                   AS sort_key
        FROM {TABLE}
        {where_sql}
        GROUP BY 1
        ORDER BY sort_key;
    """
    return _add_true_rate(_share_of_positive(run_query(sql, "fast")))


def get_unlinked_positives(start=None, end=None):
    
    where_sql = _where("FINALRESULT = 'Positive' AND (IS_LINKED = FALSE OR IS_LINKED IS NULL)",
                         _date_cond(start, end))
    sql = f"""
        SELECT
            ENCOUNTER_ID, PATIENT_ID, VISIT_DATE_D AS visit_date, AGE_BAND, SEX,
            TESTENTRYPOINT_LABEL AS entry_point, PROVIDER_ID
        FROM {TABLE}
        {where_sql}
        ORDER BY VISIT_DATE_D DESC;
    """
    df = run_query(sql, "fast")

    return df.drop_duplicates(subset="PATIENT_ID", keep="first")


def get_tb_screening(start=None, end=None):
    return get_positivity_by("TBSCREENING", order_by="positivity_rate_pct", start=start, end=end)


def get_risk_assessment(start=None, end=None):
    return get_positivity_by("ASSESSEDFORHIVRISK", order_by="positivity_rate_pct", start=start, end=end)


def get_self_test(start=None, end=None):
    return get_positivity_by("EVERHADHIVSELFTEST", order_by="positivity_rate_pct", start=start, end=end)


def get_couple_testing(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT CLIENTTESTEDAS AS category, COUNT(*) AS total_encounters
        FROM {TABLE}
        {where_sql}
        GROUP BY CLIENTTESTEDAS;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY")


def get_couple_discordance(start=None, end=None):
    where_sql = _where("CLIENTTESTEDAS = 'Couple'", _date_cond(start, end))
    sql = f"""
        SELECT COUPLEDISCORDANT AS category, COUNT(*) AS total_couples
        FROM {TABLE}
        {where_sql}
        GROUP BY COUPLEDISCORDANT;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY")


KIT_BRAND_EXCLUDE = ["DETERMINE"] 


def get_kit_brand(start=None, end=None):
    
    exclude_list = ", ".join(f"'{b}'" for b in KIT_BRAND_EXCLUDE)
    exclude_cond = f"(HIVTEST1_NAME IS NULL OR UPPER(TRIM(HIVTEST1_NAME)) NOT IN ({exclude_list}))"
    where_sql = _where(exclude_cond, _date_cond(start, end))
    sql = f"""
        SELECT INITCAP(TRIM(HIVTEST1_NAME)) AS category, COUNT(*) AS total_used
        FROM {TABLE}
        {where_sql}
        GROUP BY INITCAP(TRIM(HIVTEST1_NAME))
        ORDER BY total_used DESC;
    """
    return _fillna_category(run_query(sql, "fast"), "CATEGORY")


def get_confirmatory_completion(start=None, end=None):
    where_sql = _where("FINALRESULT = 'Positive'", _date_cond(start, end))
    sql = f"""
        SELECT
            COUNT(*)                                                     AS total_positive,
            SUM(CASE WHEN HIVTEST2_RESULT IS NOT NULL THEN 1 ELSE 0 END) AS has_test2_result,
            SUM(CASE WHEN HIVTEST3_RESULT IS NOT NULL THEN 1 ELSE 0 END) AS has_test3_result
        FROM {TABLE}
        {where_sql};
    """
    return run_query(sql, "fast")


def get_algorithm_flags(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            SUM(CASE WHEN FLAG_INVALID_POSITIVE_ALGORITHM = TRUE THEN 1 ELSE 0 END) AS invalid_positive_algorithm,
            SUM(CASE WHEN FLAG_INVALID_NEGATIVE_ALGORITHM = TRUE THEN 1 ELSE 0 END) AS invalid_negative_algorithm,
            SUM(CASE WHEN FLAG_DISCORDANT_CONFIRMATORY    = TRUE THEN 1 ELSE 0 END) AS discordant_confirmatory,
            SUM(CASE WHEN FLAG_EXPIRED_KIT_USED           = TRUE THEN 1 ELSE 0 END) AS expired_kit_used
        FROM {TABLE}
        {where_sql};
    """
    return run_query(sql, "fast")


FLAG_COLUMNS = [
    ("FLAG_SAME_DAY_DUPLICATE", "Same-day duplicate test"),
    ("FLAG_MISSING_TEST_RESULT", "Missing test result"),
    ("FLAG_INVALID_POSITIVE_ALGORITHM", "Invalid positive algorithm"),
    ("FLAG_INVALID_NEGATIVE_ALGORITHM", "Invalid negative algorithm"),
    ("FLAG_DISCORDANT_CONFIRMATORY", "Discordant confirmatory result"),
    ("FLAG_UNLINKED_POSITIVE_CLIENT", "Unlinked positive client"),
    ("FLAG_NEGATIVE_CLIENT_LINKED", "Negative client incorrectly linked"),
    ("FLAG_LINKAGE_MISMATCH", "Linkage mismatch"),
    ("FLAG_RESULT_NOT_GIVEN", "Result not given"),
    ("FLAG_GENDER_DISPARITY_ACROSS_VISITS", "Gender disparity across visits"),
    ("FLAG_AGE_DISPARITY_ACROSS_VISITS", "Age disparity across visits"),
    ("FLAG_EXPIRED_KIT_USED", "Expired kit used"),
    ("FLAG_COUPLE_FIELD_INCONSISTENCY", "Couple field inconsistency"),
    ("FLAG_INVALID_VISIT_DATE", "Invalid visit date"),
]

_ANY_FLAG_EXPR = " OR ".join(col for col, _ in FLAG_COLUMNS)


def get_flag_summary(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    select_list = ",\n            ".join(
        f"SUM(CASE WHEN {col} = TRUE THEN 1 ELSE 0 END) AS {col}" for col, _ in FLAG_COLUMNS
    )
    sql = f"""
        SELECT
            {select_list},
            COUNT(*) AS total_encounters
        FROM {TABLE}
        {where_sql};
    """
    return run_query(sql, "fast")


def get_flag_overview(start=None, end=None):
    """Single-row 'any flag raised' summary for the period — used on the
    Overview page; the per-flag breakdown lives in get_flag_summary()."""
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            COUNT(*)                                           AS total_encounters,
            SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)   AS encounters_with_any_flag,
            ROUND(100.0 * SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                     AS flag_rate_pct
        FROM {TABLE}
        {where_sql};
    """
    return run_query(sql, "fast")


def get_flag_monthly_trend(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            DATE_TRUNC('MONTH', VISIT_DATE_D)                            AS visit_month,
            COUNT(*)                                                     AS total_encounters,
            SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)            AS encounters_with_any_flag,
            ROUND(100.0 * SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                              AS flag_rate_pct
        FROM {TABLE}
        {where_sql}
        GROUP BY DATE_TRUNC('MONTH', VISIT_DATE_D)
        ORDER BY 1;
    """
    return run_query(sql, "fast")


def get_flag_type_monthly_trend(start=None, end=None):
    """Each of the 14 QA flags, counted by month — the evidence behind explaining
    a spike in the overall 'any flag raised' rate: which specific flag type(s)
    actually drove it, rather than just 'something went up'. Returned already
    unpivoted to (month, flag label, count) so it plugs straight into the same
    share-of-slice-vs-share-of-whole composition technique used for the trend
    spikes."""
    where_sql = _where(_date_cond(start, end))
    select_list = ",\n            ".join(
        f"SUM(CASE WHEN {col} = TRUE THEN 1 ELSE 0 END) AS {col}" for col, _ in FLAG_COLUMNS
    )
    sql = f"""
        SELECT
            DATE_TRUNC('MONTH', VISIT_DATE_D) AS visit_month,
            {select_list}
        FROM {TABLE}
        {where_sql}
        GROUP BY DATE_TRUNC('MONTH', VISIT_DATE_D)
        ORDER BY 1;
    """
    wide = run_query(sql, "fast")
    rows = []
    for _, row in wide.iterrows():
        for col, label in FLAG_COLUMNS:
            rows.append({"VISIT_MONTH": row["VISIT_MONTH"], "CATEGORY": label, "COUNT": float(row[col])})
    return pd.DataFrame(rows)


def get_provider_performance(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            PROVIDER_ID,
            COUNT(*)                                                     AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)    AS total_positive,
            SUM(CASE WHEN FINALRESULT = 'Positive' AND IS_LINKED = TRUE THEN 1 ELSE 0 END) AS positives_linked,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' AND IS_LINKED = TRUE THEN 1 ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END), 0), 2) AS linkage_rate_pct,
            SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)            AS encounters_with_any_flag,
            ROUND(100.0 * SUM(CASE WHEN {_ANY_FLAG_EXPR} THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                              AS flag_rate_pct
        FROM {TABLE}
        {where_sql}
        GROUP BY PROVIDER_ID
        ORDER BY total_tested DESC;
    """
    
    return _share_of_positive(run_query(sql, "fast"))

# 4b. SPIKE DEEP-DIVE RENDERING

def render_spike_block(spike, metric_label, share_col, share_of, chart_col, chart_y_title,
                          concentration_col, avg_reference, dimension_mixes, flag_trend, avg_flag_rate, color):
    month = spike["VISIT_MONTH"]
    direction = "spike" if spike["Z"] > 0 else "dip"
    st.markdown(f'<h4>{pd.Timestamp(month):%B %Y} — {metric_label} {direction}</h4>', unsafe_allow_html=True)

    #1. THE VISUAL:
    daily = get_daily_breakdown_for_month(month)
    daily_pattern_chart(daily, avg_reference, chart_col,
                          f"Daily pattern within {pd.Timestamp(month):%B %Y}", chart_y_title, color)
    pattern_text, concentrated, top_dates, top_date_values = insight_daily_pattern(daily, concentration_col, share_of)
    note(pattern_text)

    daily_strategy = None
    if concentrated and top_date_values:
        strategy_df = get_strategy_mix_for_dates(top_date_values)
        strategy_note = insight_daily_strategy(strategy_df, top_dates)
        if strategy_note:
            note(strategy_note)
            daily_strategy = strategy_df.loc[strategy_df["TOTAL_POSITIVE"].idxmax(), "CATEGORY"]

    drivers = analyze_spike_factors(month, share_col, dimension_mixes, require_threshold=False, top_n=8)
    if drivers:
        comp_df = pd.DataFrame([{"CATEGORY": f"{d['dimension']}: {d['category']}", "DELTA_PTS": d["delta_pts"]}
                                  for d in drivers])
        bar_chart(comp_df, "CATEGORY", "DELTA_PTS", "Share of that month vs. usual share, by factor",
                   color=CATEGORICAL[7], horizontal=True, height=280,
                   x_title="Percentage points above usual share")
    note(insight_composition_chart(drivers, share_of))

    month_flag_row = flag_trend[flag_trend["VISIT_MONTH"] == pd.Timestamp(month)]
    month_flag_rate = float(month_flag_row["FLAG_RATE_PCT"].iloc[0]) if not month_flag_row.empty else None
    quality_flag = quality_caveat_for_month(month, flag_trend)
    if month_flag_rate is not None:
        c1, c2 = st.columns(2)
        kpi_tile(c1, "Flag rate this month", f"{month_flag_rate:.1f}%",
                  color=status_color(month_flag_rate, avg_flag_rate, avg_flag_rate * 1.5, higher_is_better=False))
        kpi_tile(c2, "Period average flag rate", f"{avg_flag_rate:.1f}%")

    recommend(recommend_for_spike(month, drivers, quality_flag, concentrated, top_dates,
                                    daily_strategy=daily_strategy))
    st.divider()


def render_flag_rate_spikes(flag_trend, start_date, end_date, z_thresh=1.3):
    """Same visual -> finding pattern as the trend spikes, applied to the
    data-quality flag-rate trend: names WHICH of the 14 flag types actually
    drove a spike month, instead of leaving 'the flag rate went up' unexplained.
    Computes every block BEFORE rendering the section title, so a spike month
    that turns out to have no flag-type breakdown to show (e.g. no matching
    month in the flag-type data) never leaves a heading with nothing under it."""
    spikes = detect_spikes(flag_trend, "FLAG_RATE_PCT", z_thresh=z_thresh, max_n=2)
    if not spikes:
        return
    flag_type_trend = get_flag_type_monthly_trend(start_date, end_date)
    blocks = []
    for spike in spikes:
        month = spike["VISIT_MONTH"]
        table = composition_delta_table(flag_type_trend, pd.Timestamp(month), "VISIT_MONTH", "COUNT",
                                          cat_col="CATEGORY")
        if not table.empty:
            blocks.append((month, table))
    if not blocks:
        return

    st.markdown('<h4>What drove the flag-rate spikes</h4>', unsafe_allow_html=True)
    for month, table in blocks:
        month_str = f"{pd.Timestamp(month):%B %Y}"
        st.markdown(f'<h5>{month_str}</h5>', unsafe_allow_html=True)
        bar_chart(table, "CATEGORY", "DELTA_PTS", f"Flag type share of {month_str} vs. usual share",
                   color=CATEGORICAL[7], horizontal=True, height=320,
                   x_title="Percentage points above usual share")
        top = table.iloc[0]
        if top["DELTA_PTS"] >= 5:
            note(f"<b>{top['CATEGORY']}</b> flags drove this spike — {top['SLICE_SHARE']:.0f}% of {month_str}'s "
                 f"flagged encounters vs. {top['OVERALL_SHARE']:.0f}% typically, a {top['DELTA_PTS']:.0f}-point "
                 f"jump. Worth a targeted QA review of that specific check for that month.")
        else:
            note(f"No single flag type stands out for {month_str} — the largest shift was "
                 f"<b>{top['CATEGORY']}</b> at just {top['DELTA_PTS']:.0f} points above its usual share, so the "
                 f"rise looks spread across multiple flag types rather than one dominant cause.")


# 5. VISUALS  
st.markdown('<div class="app-title" style="font-size:32px;">Kakamega HIV Testing Services — Analysis Dashboard</div>',
            unsafe_allow_html=True)

try:
    get_connection()
except Exception as e:  # noqa: BLE001
    st.error(
        "Couldn't connect to Snowflake. Check your `.env` file has the variables listed "
        "at the top of this file (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE, "
        "SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, and one of SNOWFLAKE_PRIVATE_KEY_PATH / "
        "SNOWFLAKE_PASSWORD / SNOWFLAKE_AUTHENTICATOR) — rename your existing keys to match, "
        "or edit the os.environ[...] lines in the CONNECTION section above."
    )
    st.exception(e)
    st.stop()

try:
    _date_bounds_probe = get_date_bounds().iloc[0]
except Exception as e:  # noqa: BLE001
    st.error(
        "Lost the Snowflake connection while loading the date range — this usually means the "
        "session/auth token expired mid-use. Click below to reconnect."
    )
    st.exception(e)
    if st.button("🔄 Reconnect to Snowflake"):
        get_connection.clear()
        st.cache_data.clear()
        st.rerun()
    st.stop()
    
with st.sidebar:
    st.markdown(
        f'<div style="font-family:Poppins,sans-serif; font-weight:700; font-size:20px; '
        f'color:{NAVY}; margin-bottom:4px;">Kakamega HTS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Analytics navigation")
    section = st.radio("Section", SECTIONS, label_visibility="collapsed")

    st.markdown("---")
    st.caption("Date range")
    bounds = _date_bounds_probe
    min_d = pd.to_datetime(bounds.MIN_DATE).date()
    max_d = pd.to_datetime(bounds.MAX_DATE).date()
    picked = st.date_input(
        "Visit date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
        label_visibility="collapsed",
    )
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start_date, end_date = picked
    elif isinstance(picked, (tuple, list)) and len(picked) == 1:
        start_date, end_date = picked[0], max_d
    else:
        start_date, end_date = min_d, max_d
    st.caption(f"{start_date} → {end_date}")

    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        get_connection.clear()
        st.rerun()
    st.caption("Cached results are re-used for up to 30 minutes per query; use Refresh to force a re-pull "
               "(also reconnects to Snowflake if your session has expired).")

# Overview 
if section == "Overview":
    kpis = get_kpis(start_date, end_date).iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_tile(c1, "Total tested", f"{int(kpis.TOTAL_TESTED):,}")
    kpi_tile(c2, "Unique clients", f"{int(kpis.UNIQUE_CLIENTS):,}")
    kpi_tile(c3, "Positivity", f"{int(kpis.TOTAL_POSITIVE):,} | {kpis.POSITIVITY_RATE_PCT:.2f}%")
    kpi_tile(c4, "Linkage rate", f"{kpis.LINKAGE_RATE_PCT:.1f}%",
              color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75))
    kpi_tile(c5, "Result-given rate", f"{kpis.RESULT_GIVEN_RATE_PCT:.1f}%",
              color=status_color(kpis.RESULT_GIVEN_RATE_PCT, 95, 90))

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown('<h3>At a glance across every section</h3>', unsafe_allow_html=True)

    row_a = st.columns(3)
    with row_a[0].container(height=320, border=True):
        st.markdown("**Burden & Positivity**")
        df = get_positivity_by("SEX", start=start_date, end=end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positive share by sex", color=CATEGORICAL[0],
                   pct=True, height=210)
    with row_a[1].container(height=320, border=True):
        st.markdown("**Testing Performance**")
        df = get_strategy_yield(start_date, end_date).head(5)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Top strategies by positive share", color=CATEGORICAL[1],
                   horizontal=True, pct=True, height=210)
    with row_a[2].container(height=320, border=True):
        st.markdown("**Care Cascade**")
        unlinked = get_unlinked_positives(start_date, end_date)
        sub1, sub2 = st.columns(2)
        kpi_tile(sub1, "Linkage rate", f"{kpis.LINKAGE_RATE_PCT:.1f}%",
                  color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75))
        kpi_tile(sub2, "Unlinked positives", f"{len(unlinked)}",
                  color=status_color(len(unlinked), 0, 5, higher_is_better=False))
        st.caption("Clients flagged positive but not (yet) linked to care in this period.")

    row_b = st.columns(3)
    with row_b[0].container(height=320, border=True):
        st.markdown("**Clinical Correlations**")
        risk = get_risk_assessment(start_date, end_date)
        zero_positive_when_assessed = (
            risk.loc[risk["CATEGORY"] == True, "TOTAL_POSITIVE"].sum() == 0  # noqa: E712
            if True in risk["CATEGORY"].values else False
        )
        if zero_positive_when_assessed:
            st.markdown(
                f'<div style="color:{STATUS["critical"]}; font-size:13px;">'
                f'⚠️ Zero positives among risk-assessed clients — flagged for medical review.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Risk-assessment vs. TB-screening vs. self-test correlations — see full section.")
       
        bar_chart(risk, "CATEGORY", "POSITIVITY_RATE_PCT", "Positive share by risk-assessment status",
                   color=CATEGORICAL[7], pct=True, height=170)
    with row_b[1].container(height=320, border=True):
        st.markdown("**Kit / Algorithm QA**")
        alg = get_algorithm_flags(start_date, end_date).iloc[0]
    
        kpi_tile(st.container(), "Invalid algorithms",
                  int(alg.INVALID_POSITIVE_ALGORITHM) + int(alg.INVALID_NEGATIVE_ALGORITHM),
                  color=status_color(int(alg.INVALID_POSITIVE_ALGORITHM) + int(alg.INVALID_NEGATIVE_ALGORITHM),
                                       0, 2, higher_is_better=False))
        st.caption("Combined positive + negative algorithm-application errors, this period.")
    with row_b[2].container(height=320, border=True):
        st.markdown("**Data Integrity**")
        fo = get_flag_overview(start_date, end_date).iloc[0]
        sub1, sub2 = st.columns(2)
        kpi_tile(sub1, "Encounters flagged", f"{int(fo.ENCOUNTERS_WITH_ANY_FLAG):,}")
        kpi_tile(sub2, "Flag rate", f"{fo.FLAG_RATE_PCT:.1f}%",
                  color=status_color(fo.FLAG_RATE_PCT, 2, 5, higher_is_better=False))
        st.caption("Share of encounters raising at least one of the 14 QA flags.")

    st.markdown('<h3>Provider performance snapshot</h3>', unsafe_allow_html=True)
    prov = get_provider_performance(start_date, end_date).head(5).copy()
    prov["PROVIDER_ID"] = prov["PROVIDER_ID"].str[:10] + "…"
    table_view(prov, caption="Top 5 providers by volume in the selected period — full breakdown under "
                              "Provider Performance.")

#  Testing & Care Cascade
elif section == "Testing & Care Cascade":
    kpis = get_kpis(start_date, end_date).iloc[0]

    st.subheader("Monthly testing volume and positive-share trend")
    trend = get_monthly_trend(start_date, end_date)
   
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=trend["VISIT_MONTH"], y=trend["TOTAL_TESTED"], marker_color=CATEGORICAL[0],
                 name="Total tested", secondary_y=False)
    fig.add_scatter(x=trend["VISIT_MONTH"], y=trend["POSITIVITY_RATE_PCT"], name="Share of positive cases (%)",
                     mode="lines+markers", marker_color=CATEGORICAL[7], line=dict(width=2), secondary_y=True)
    fig.update_layout(title="Monthly testing volume vs. share of positive cases", height=420, showlegend=True)
    fig = plotly_template(fig)
    fig.update_yaxes(title_text="Total tested", color=CATEGORICAL[0], secondary_y=False)
    fig.update_yaxes(title_text="Share of positive cases (%)", color=CATEGORICAL[7], secondary_y=True,
                       showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Bars (left axis) show testing volume; the line (right axis, colored to match) shows that month's "
        "share of positive cases — both axes are labeled and colored so each series reads against its own scale."
    )

    
    st.markdown('<h4>High- vs. low-positivity months: testing strategy and entry point</h4>',
                 unsafe_allow_html=True)
    high_months, low_months = split_high_low_months(trend)
    if high_months is None:
        note("Not enough months in the selected period to compare high- and low-positivity months "
             "(need at least 4).")
    else:
        strategy_mix = get_monthly_strategy_mix(start_date, end_date)
        entrypoint_mix = get_monthly_entrypoint_mix(start_date, end_date)

        
        high_table = build_month_detail_table(high_months, trend, strategy_mix, entrypoint_mix)
        low_table = build_month_detail_table(low_months, trend, strategy_mix, entrypoint_mix)
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**High-positivity months**")
            st.dataframe(high_table, hide_index=True, width="stretch")
        with t2:
            st.markdown("**Low-positivity months**")
            st.dataframe(low_table, hide_index=True, width="stretch")

        strategy_comp = compare_high_low_mix(strategy_mix, high_months, low_months)
        if not strategy_comp.empty:
            note(insight_high_low_comparison(strategy_comp, "testing strategy"))

        entrypoint_comp = compare_high_low_mix(entrypoint_mix, high_months, low_months)
        if not entrypoint_comp.empty:
            note(insight_high_low_comparison(entrypoint_comp, "entry point"))

    
    st.markdown("---")
    st.subheader("Where testing happens, and how positivity varies by setting")
    df = get_positivity_by("SETTING", start=start_date, end=end_date)
    volume_and_share_pair(df, "setting", color=CATEGORICAL[2])
    note(insight_categorical(df, "setting"))
    render_category_driver(df, get_driver_factor_mixes("SETTING", start_date, end_date))

    df = get_entrypoint_yield(start_date, end_date)
    volume_and_share_pair(df, "entry point", color=CATEGORICAL[0], horizontal=True)
    note(insight_categorical(df, "entry point"))

    st.markdown("---")
    st.subheader("How they were tested")
    df = get_strategy_yield(start_date, end_date)
    volume_and_share_pair(df, "testing strategy", color=CATEGORICAL[0], horizontal=True)
    note(insight_categorical(df, "testing strategy"))

    df = get_retest_interval(start_date, end_date)
    volume_and_share_pair(df, "time since last test", color=CATEGORICAL[0])
    note(
        "The 'No prior test / not recorded' bucket mixes true first-time testers with repeat testers whose "
        "retest interval simply wasn't captured, so it overstates first-time volume — read it as an upper "
        "bound, not an exact count. " + insight_categorical(df, "retest interval")
    )

    st.markdown("---")
    st.subheader("Who's being tested, and how positivity varies by group")
    df = get_positivity_by("SEX", start=start_date, end=end_date)
    volume_and_share_pair(df, "sex", color=CATEGORICAL[0])
    note(insight_categorical(df, "sex"))
    render_category_driver(df, get_driver_factor_mixes("SEX", start_date, end_date), always_show=True)

    df = get_positivity_by("AGE_BAND", order_by="total_tested", start=start_date, end=end_date)
    volume_and_share_pair(df, "age band", color=CATEGORICAL[0])
    note(insight_categorical(df, "age band"))
    render_category_driver(df, get_driver_factor_mixes("AGE_BAND", start_date, end_date))

    df = get_positivity_by("MARITAL_STATUS_LABEL", order_by="total_tested", start=start_date, end=end_date)
    volume_and_share_pair(df, "marital status", color=CATEGORICAL[0], horizontal=True)
    note(insight_categorical(df, "marital status"))
    render_category_driver(df, get_driver_factor_mixes("MARITAL_STATUS_LABEL", start_date, end_date))

    st.markdown("---")
    st.subheader("Care cascade — from a positive result to linkage")
    c1, c2, c3 = st.columns(3)

    GAUGE_HEIGHT = 220
    kpi_tile(c1, "Positives linked to care", f"{int(kpis.POSITIVES_LINKED)} / {int(kpis.TOTAL_POSITIVE)}",
              color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75), min_height=GAUGE_HEIGHT)
    
    with c2:
        if not pd.isna(kpis.LINKAGE_RATE_PCT):
            gauge_meter(kpis.LINKAGE_RATE_PCT, "Linkage rate vs. 95% benchmark", target=95, warn_at=75,
                         height=GAUGE_HEIGHT)
        else:
            st.caption("Linkage rate unavailable for this period.")
    kpi_tile(c3, "Result-given rate", f"{kpis.RESULT_GIVEN_RATE_PCT:.1f}%",
              color=status_color(kpis.RESULT_GIVEN_RATE_PCT, 95, 90), min_height=GAUGE_HEIGHT)
    note(insight_linkage(kpis))

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Unlinked positive clients — needs follow-up**")
    unlinked = get_unlinked_positives(start_date, end_date)
    render_unlinked_detail(unlinked)

# Clinical & Data Quality 
elif section == "Clinical & Data Quality":
    st.subheader("Clinical correlations")
    risk = get_risk_assessment(start_date, end_date)
    volume_and_share_pair(risk, "risk-assessment status", color=CATEGORICAL[7], always_show_share=True)
    note(insight_categorical(risk, "risk-assessment status"))
    render_category_driver(risk, get_driver_factor_mixes("ASSESSEDFORHIVRISK", start_date, end_date))

    df = get_tb_screening(start_date, end_date)
    volume_and_share_pair(df, "TB screening status", color=CATEGORICAL[1], horizontal=True)
    note(insight_categorical(df, "TB screening status"))
    render_category_driver(df, get_driver_factor_mixes("TBSCREENING", start_date, end_date))

    df = get_self_test(start_date, end_date)
    volume_and_share_pair(df, "self-test history", color=CATEGORICAL[2])
    note(insight_categorical(df, "self-test history"))
    render_category_driver(df, get_driver_factor_mixes("EVERHADHIVSELFTEST", start_date, end_date))

    couple_df = get_couple_testing(start_date, end_date)
    disc_df = get_couple_discordance(start_date, end_date)
    left, right = st.columns(2)
    with left:
        stacked_share_bar(couple_df, "CATEGORY", "TOTAL_ENCOUNTERS", "Individual vs. couple testing",
                            colors=[CATEGORICAL[4], CATEGORICAL[2], CATEGORICAL[3]])
    with right:
        stacked_share_bar(disc_df, "CATEGORY", "TOTAL_COUPLES", "Discordance among couples tested",
                            colors=[CATEGORICAL[3], CATEGORICAL[4], CATEGORICAL[2]])
    note(insight_couple(couple_df, disc_df))

    st.markdown("---")
    st.subheader("Kit and testing-algorithm QA")
    left, right = st.columns([2, 1])
    with left:
        df = get_kit_brand(start_date, end_date)
        bar_chart(df, "CATEGORY", "TOTAL_USED", "Test kit brand distribution", color=CATEGORICAL[0])
        st.caption(f"Records naming an unauthorized kit brand ({', '.join(KIT_BRAND_EXCLUDE).title()}) are "
                    "excluded here as a data-quality issue, not shown as a normal category.")
    with right:
        conf = get_confirmatory_completion(start_date, end_date).iloc[0]
        kpi_tile(right, "Positives with 2nd algorithm test",
                  f"{int(conf.HAS_TEST2_RESULT)} / {int(conf.TOTAL_POSITIVE)}")
        kpi_tile(right, "Positives with 3rd algorithm test",
                  f"{int(conf.HAS_TEST3_RESULT)} / {int(conf.TOTAL_POSITIVE)}")
    note(insight_confirmatory(conf))

    alg = get_algorithm_flags(start_date, end_date).iloc[0]
    
    alg_cards = [
        ("Expired kit used", int(alg.EXPIRED_KIT_USED), 0, 0),
        ("Invalid positive algorithm", int(alg.INVALID_POSITIVE_ALGORITHM), 0, 2),
        ("Invalid negative algorithm", int(alg.INVALID_NEGATIVE_ALGORITHM), 0, 2),
    ]
    nonzero_cards = [c for c in alg_cards if c[1] != 0]
    if nonzero_cards:
        for col, (label, value, good_at, warn_at) in zip(st.columns(len(nonzero_cards)), nonzero_cards):
            kpi_tile(col, label, value, color=status_color(value, good_at, warn_at, higher_is_better=False))
    else:
        st.caption("No expired-kit or invalid-algorithm cases in this period.")
    note(insight_algorithm_flags(alg))

    st.markdown("---")
    st.subheader("Data integrity — the 14 QA flags")
    fo = get_flag_overview(start_date, end_date).iloc[0]
    c1, c2, c3 = st.columns(3)
    kpi_tile(c1, "Total encounters", f"{int(fo.TOTAL_ENCOUNTERS):,}")
    kpi_tile(c2, "Encounters flagged", f"{int(fo.ENCOUNTERS_WITH_ANY_FLAG):,}")
    kpi_tile(c3, "Flag rate", f"{fo.FLAG_RATE_PCT:.1f}%",
              color=status_color(fo.FLAG_RATE_PCT, 2, 5, higher_is_better=False))

    st.markdown("&nbsp;", unsafe_allow_html=True)
    flags = get_flag_summary(start_date, end_date).iloc[0]
    flag_df = pd.DataFrame(
        [{"CATEGORY": label, "COUNT": int(flags[col])} for col, label in FLAG_COLUMNS]
    ).sort_values("COUNT", ascending=False)
    bar_chart(flag_df, "CATEGORY", "COUNT", "Data-quality flag counts (all 14 flags)",
               color=CATEGORICAL[7], horizontal=True)
    note(insight_leading_flag(flag_df, int(fo.TOTAL_ENCOUNTERS)))

    flag_type_trend = get_flag_type_monthly_trend(start_date, end_date)
    if not flag_type_trend.empty:
        pivot = flag_type_trend.pivot(index="CATEGORY", columns="VISIT_MONTH", values="COUNT").fillna(0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]  # busiest flag types on top
        heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[pd.Timestamp(c).strftime("%b %Y") for c in pivot.columns],
            y=list(pivot.index),
            colorscale=[[0, "#eef3fb"], [1, CATEGORICAL[7]]],
            colorbar=dict(title="Count", tickfont=dict(color=INK["muted"])),
            hovertemplate="%{y}, %{x}: %{z:.0f}<extra></extra>",
        ))
        heat.update_layout(title="Which flag types fired, month by month", height=max(320, 26 * len(pivot.index)))
        heat = plotly_template(heat)
        heat.update_xaxes(showgrid=False)
        heat.update_yaxes(showgrid=False, autorange="reversed")
        st.plotly_chart(heat, width="stretch")
        note("Darker cells mark months where a specific flag type fired more often — use this alongside the "
             "spike breakdown below to see at a glance whether a rise was one flag type or spread across several.")

    trend = get_flag_monthly_trend(start_date, end_date)
    render_flag_rate_spikes(trend, start_date, end_date)

# Provider Performance 
elif section == "Provider Performance":
    st.subheader("Volume, positive share, and data quality by provider")
    prov = get_provider_performance(start_date, end_date)
    prov_display = prov.copy()
    prov_display["PROVIDER_ID"] = prov_display["PROVIDER_ID"].str[:10] + "…"

    RELIABILITY_N = 30
    confidence = (prov["TOTAL_TESTED"] / RELIABILITY_N).clip(upper=1.0)
    sized_flag_rate = prov["FLAG_RATE_PCT"].fillna(0) * confidence

    fig = go.Figure()
    fig.add_scatter(
        x=prov["TOTAL_TESTED"], y=prov["POSITIVITY_RATE_PCT"], mode="markers",
        marker=dict(
            size=(sized_flag_rate + 4) * 2.5,
            color=CATEGORICAL[0],
            line=dict(width=1, color=INK["primary"]),
        ),
        text=prov_display["PROVIDER_ID"],
        customdata=prov["FLAG_RATE_PCT"].fillna(0),
        hovertemplate="Provider %{text}<br>Tested: %{x}<br>Share of all positives: %{y:.2f}%<br>"
                       "Flag rate: %{customdata:.1f}%<extra></extra>",
    )
    fig.update_layout(title="Provider volume vs. positive share (bubble size = flag rate)", height=420)
    fig = plotly_template(fig, y_title="Share of positive cases (%)")

    fig.update_xaxes(title="Total tested", rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    table_view(
        prov_display,
        caption="Provider IDs are hashed — map back to real facility names internally if you keep that key. "
                "Sort by FLAG_RATE_PCT or LINKAGE_RATE_PCT to spot outlier sites.",
    )
    note(insight_provider_outliers(prov))

    st.markdown("---")
    st.subheader("Unlinked positive clients, by provider — needs follow-up")
    unlinked = get_unlinked_positives(start_date, end_date)
    render_unlinked_detail(unlinked)