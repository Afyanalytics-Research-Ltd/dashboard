"""
Kakamega HIV Testing Services (HTS) — hybrid analysis dashboard.


"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os

import pandas as pd
import plotly.graph_objects as go
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv

# ============================================================================
# 2. CONNECTION  (reads your .env, caches the Snowflake connection + queries)
# ============================================================================
load_dotenv()

TABLE = "hospitals.staging.HIV_HTS_STAGING"

REQUIRED_ENV = [
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
]


@st.cache_resource
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


@st.cache_data(ttl=300)
def _run_fast(sql):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
        return _coerce_decimals(cur.fetch_pandas_all())


@st.cache_data(ttl=1800)
def _run_slow(sql):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
        return _coerce_decimals(cur.fetch_pandas_all())


def run_query(sql, speed="slow"):
    """speed='fast' -> 5 min cache (small lookups); 'slow' -> 30 min cache (full-table rollups)."""
    return _run_fast(sql) if speed == "fast" else _run_slow(sql)


# ============================================================================
# 3. STYLE / THEME  (navy + teal brand theme, dataviz house palette for charts)
# ============================================================================

# --- chart palette: unchanged from the validated dataviz default (fixed
#     categorical order, single-hue sequential, reserved status colors) ---
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = "#2a78d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# --- brand theme: navy ink + teal accent + white surfaces ---
NAVY = "#0b2545"
ACCENT = CATEGORICAL[2]          # teal/aqua — reused as the UI accent (chrome, not a data series)
ACCENT_DARK = "#158f66"
INK = {"primary": "#132436", "secondary": "#4b5b6b", "muted": "#8a97a3"}
SURFACE = "#ffffff"
PAGE_BG = "#f7f9fb"
GRIDLINE = "#e3e8ec"

SECTIONS = [
    "Overview", "Testing & Care Cascade", "Clinical & Data Quality", "Provider Performance",
    "Spike Deep-Dive",
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


def kpi_tile(col, label, value, help_text=None, color=None, small=False):
    """Fixed-height metric card — every card renders the same label / value /
    help structure so a row of cards always lines up, regardless of how long
    any individual value or help string is. Pass small=True for a value
    string too long to read comfortably at the default 26px (e.g. a date
    range) — the card stays the same height either way."""
    value_size = "16px" if small else "26px"
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
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


# ============================================================================
# 3b. DESCRIPTIVE-ANALYSIS HELPERS
#     Every function here reads the same dataframe already fetched for its
#     chart and turns it into a plain-language sentence — computed fresh
#     against whatever date range is selected, never a hardcoded number.
# ============================================================================

def insight_categorical(df, label, cat_col="CATEGORY", rate_col="POSITIVITY_RATE_PCT",
                          n_col="TOTAL_TESTED", small_n=100, unit="Positivity"):
    d = df.dropna(subset=[rate_col])
    if len(d) < 2:
        return f"Not enough categories in the selected period to compare {label}."
    top = d.loc[d[rate_col].idxmax()]
    bottom = d.loc[d[rate_col].idxmin()]
    if bottom[rate_col] > 0:
        ratio_txt = f", roughly {top[rate_col] / bottom[rate_col]:.1f}x higher"
    else:
        ratio_txt = ""
    caveats = [f"**{row[cat_col]}** rests on a small sample (n={int(row[n_col])})"
               for _, row in pd.DataFrame([top, bottom]).iterrows() if row[n_col] < small_n]
    caveat_txt = f" Treat with caution — {' and '.join(caveats)}." if caveats else ""
    return (f"{unit} is highest among <b>{top[cat_col]}</b> at {top[rate_col]:.1f}% and lowest among "
            f"<b>{bottom[cat_col]}</b> at {bottom[rate_col]:.1f}%{ratio_txt}.{caveat_txt}")


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
    disc_row = discordance_df[discordance_df["CATEGORY"] == True]  # noqa: E712
    disc_n = int(disc_row["TOTAL_COUPLES"].iloc[0]) if not disc_row.empty else 0
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


def explain_month_composition(mix_df, month, share_col="TOTAL_TESTED", month_col="VISIT_MONTH",
                                cat_col="CATEGORY", min_delta_pts=5, min_overall_count=0):
    """Compares each category's share of ONE month's volume (or positives) against
    its share of the whole selected period, and returns whichever category is most
    over-represented that month — the composition change most likely contributing to
    a spike. min_overall_count drops thin categories (e.g. a low-volume provider)
    whose share can swing wildly on noise alone."""
    month_ts = pd.Timestamp(month)
    overall_counts = mix_df.groupby(cat_col)[share_col].sum()
    valid_cats = overall_counts[overall_counts >= min_overall_count].index
    month_slice = mix_df[(mix_df[month_col] == month_ts) & (mix_df[cat_col].isin(valid_cats))]
    if month_slice.empty or month_slice[share_col].sum() == 0:
        return None
    month_share = month_slice.groupby(cat_col)[share_col].sum()
    month_share = month_share / month_share.sum()
    overall_share = overall_counts.reindex(valid_cats)
    overall_share = overall_share / overall_share.sum()
    delta = (month_share - overall_share.reindex(month_share.index).fillna(0)).sort_values(ascending=False)
    if delta.empty or delta.iloc[0] * 100 < min_delta_pts:
        return None
    top_cat = delta.index[0]
    return {
        "category": top_cat,
        "month_share": month_share[top_cat] * 100,
        "overall_share": overall_share.get(top_cat, 0) * 100,
        "delta_pts": delta.iloc[0] * 100,
    }


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
                 "sustained shift."), False, ""
    k = min(3, len(d))
    top = d.nlargest(k, count_col)
    top_share = 100.0 * top[count_col].sum() / d[count_col].sum()
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
    return text, concentrated, top_dates


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


def recommend_for_spike(month, drivers, quality_flag, concentrated, top_dates, threshold=5):
    """Synthesizes all three angles — daily concentration, factor composition, and
    data-quality context — into one actionable recommendation, so it never rests on
    just one lens."""
    month_str = f"{pd.Timestamp(month):%B %Y}"
    lines = []
    if quality_flag:
        lines.append(f"the data-quality flag rate also jumped in {month_str} — rule out duplicate or "
                       "backfilled records for that month before treating this as a genuine trend")
    if concentrated and top_dates:
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


# ============================================================================
# 4. SQL QUERY FUNCTIONS  (one per metric, fully qualified against TABLE,
#    every function takes optional start/end to apply the sidebar date filter)
# ============================================================================

def _date_cond(start, end):
    if start and end:
        return f"VISIT_DATE_D BETWEEN '{start}' AND '{end}'"
    return None


def _where(*conditions):
    conds = [c for c in conditions if c]
    return ("WHERE " + " AND ".join(conds)) if conds else ""


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
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                             AS positivity_rate_pct
        FROM {TABLE}
        {where_sql}
        GROUP BY DATE_TRUNC('MONTH', VISIT_DATE_D)
        ORDER BY 1;
    """
    return run_query(sql, "fast")


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
    return run_query(sql, "fast")


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


def get_positivity_by(column, order_by="total_tested", start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            {column}                                                    AS category,
            COUNT(*)                                                    AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)   AS total_positive,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                             AS positivity_rate_pct
        FROM {TABLE}
        {where_sql}
        GROUP BY {column}
        ORDER BY {order_by} DESC;
    """
    return run_query(sql, "fast")


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
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                              AS positivity_rate_pct,
            MIN(COALESCE(MONTHSSINCELASTTEST_NUM, -1))                   AS sort_key
        FROM {TABLE}
        {where_sql}
        GROUP BY 1
        ORDER BY sort_key;
    """
    return run_query(sql, "fast")


def get_unlinked_positives(start=None, end=None):
    where_sql = _where("FLAG_UNLINKED_POSITIVE_CLIENT = TRUE", _date_cond(start, end))
    sql = f"""
        SELECT
            ENCOUNTER_ID, VISIT_DATE_D AS visit_date, AGE_BAND, SEX,
            TESTENTRYPOINT_LABEL AS entry_point, PROVIDER_ID
        FROM {TABLE}
        {where_sql}
        ORDER BY VISIT_DATE_D DESC;
    """
    return run_query(sql, "fast")


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
    return run_query(sql, "fast")


def get_couple_discordance(start=None, end=None):
    where_sql = _where("CLIENTTESTEDAS = 'Couple'", _date_cond(start, end))
    sql = f"""
        SELECT COUPLEDISCORDANT AS category, COUNT(*) AS total_couples
        FROM {TABLE}
        {where_sql}
        GROUP BY COUPLEDISCORDANT;
    """
    return run_query(sql, "fast")


def get_kit_brand(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT HIVTEST1_NAME AS category, COUNT(*) AS total_used
        FROM {TABLE}
        {where_sql}
        GROUP BY HIVTEST1_NAME
        ORDER BY total_used DESC;
    """
    return run_query(sql, "fast")


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


def get_provider_performance(start=None, end=None):
    where_sql = _where(_date_cond(start, end))
    sql = f"""
        SELECT
            PROVIDER_ID,
            COUNT(*)                                                     AS total_tested,
            SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)    AS total_positive,
            ROUND(100.0 * SUM(CASE WHEN FINALRESULT = 'Positive' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0), 2)                              AS positivity_rate_pct,
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
    return run_query(sql, "fast")


# ============================================================================
# 4b. SPIKE DEEP-DIVE RENDERING
#     One function that runs the full chart -> finding -> cause -> recommendation
#     flow for a single anomalous month, so the Spike Deep-Dive section is just a
#     loop calling this once per detected spike.
# ============================================================================

def render_spike_block(spike, metric_label, share_col, share_of, chart_col, chart_y_title,
                          concentration_col, avg_reference, dimension_mixes, flag_trend, avg_flag_rate, color):
    month = spike["VISIT_MONTH"]
    direction = "spike" if spike["Z"] > 0 else "dip"
    st.markdown(f'<h4>{pd.Timestamp(month):%B %Y} — {metric_label} {direction}</h4>', unsafe_allow_html=True)

    # 1. THE VISUAL: day-by-day pattern within the month itself (new query — this is
    #    the actual evidence, not an inference from the monthly total alone).
    daily = get_daily_breakdown_for_month(month)
    daily_pattern_chart(daily, avg_reference, chart_col,
                          f"Daily pattern within {pd.Timestamp(month):%B %Y}", chart_y_title, color)
    pattern_text, concentrated, top_dates = insight_daily_pattern(daily, concentration_col, share_of)
    note(pattern_text)

    # 2. THE VISUAL: every tracked factor's share-of-month vs. its usual share,
    #    ranked — charted even when nothing crosses the threshold, so "nothing here
    #    explains it" is something the audience can see, not just be told.
    drivers = analyze_spike_factors(month, share_col, dimension_mixes, require_threshold=False, top_n=8)
    if drivers:
        comp_df = pd.DataFrame([{"CATEGORY": f"{d['dimension']}: {d['category']}", "DELTA_PTS": d["delta_pts"]}
                                  for d in drivers])
        bar_chart(comp_df, "CATEGORY", "DELTA_PTS", "Share of that month vs. usual share, by factor",
                   color=CATEGORICAL[7], horizontal=True, height=280,
                   x_title="Percentage points above usual share")
    note(insight_composition_chart(drivers, share_of))

    # 3. THE VISUAL: this month's data-quality flag rate next to the period average —
    #    rules in or out a data-entry explanation before trusting the other two.
    month_flag_row = flag_trend[flag_trend["VISIT_MONTH"] == pd.Timestamp(month)]
    month_flag_rate = float(month_flag_row["FLAG_RATE_PCT"].iloc[0]) if not month_flag_row.empty else None
    quality_flag = quality_caveat_for_month(month, flag_trend)
    c1, c2 = st.columns(2)
    kpi_tile(c1, "Flag rate this month", f"{month_flag_rate:.1f}%" if month_flag_rate is not None else "n/a",
              color=(status_color(month_flag_rate, avg_flag_rate, avg_flag_rate * 1.5, higher_is_better=False)
                      if month_flag_rate is not None else None))
    kpi_tile(c2, "Period average flag rate", f"{avg_flag_rate:.1f}%")

    # 4. THE RECOMMENDATION: synthesizes all three angles above into one action.
    recommend(recommend_for_spike(month, drivers, quality_flag, concentrated, top_dates))
    st.divider()


# ============================================================================
# 5. VISUALS  (sidebar navigation + date filter, 4 sections, each chart
#    followed by a computed descriptive-analysis note)
# ============================================================================
st.markdown('<div class="app-title" style="font-size:32px;">Kakamega HIV Testing Services — Analysis Dashboard</div>',
            unsafe_allow_html=True)
st.caption("Source: hospitals.staging.HIV_HTS_STAGING · Validated against the original CSV extract")

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

# ---- Sidebar: brand, section navigation, date filter, refresh ----
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
    bounds = get_date_bounds().iloc[0]
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
        st.rerun()
    st.caption("Cached results are re-used for up to 30 minutes per query; use Refresh to force a re-pull.")

# ---- Overview ----
if section == "Overview":
    kpis = get_kpis(start_date, end_date).iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_tile(c1, "Total tested", f"{int(kpis.TOTAL_TESTED):,}")
    kpi_tile(c2, "Unique clients", f"{int(kpis.UNIQUE_CLIENTS):,}")
    kpi_tile(c3, "Positivity rate", f"{kpis.POSITIVITY_RATE_PCT:.2f}%")
    kpi_tile(c4, "Linkage rate", f"{kpis.LINKAGE_RATE_PCT:.1f}%",
              color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75))
    kpi_tile(c5, "Result-given rate", f"{kpis.RESULT_GIVEN_RATE_PCT:.1f}%",
              color=status_color(kpis.RESULT_GIVEN_RATE_PCT, 95, 90))
    kpi_tile(c6, "Selected period", f"{start_date:%b %Y} – {end_date:%b %Y}", small=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    trend = get_monthly_trend(start_date, end_date)
    fig = go.Figure()
    fig.add_bar(x=trend["VISIT_MONTH"], y=trend["TOTAL_TESTED"], name="Total tested",
                marker_color=CATEGORICAL[0], opacity=0.35, yaxis="y")
    fig.add_scatter(x=trend["VISIT_MONTH"], y=trend["POSITIVITY_RATE_PCT"], name="Positivity rate (%)",
                     mode="lines+markers", marker_color=CATEGORICAL[7], line=dict(width=2), yaxis="y2")
    fig.update_layout(
        title="Monthly testing volume and positivity rate",
        height=380,
        yaxis=dict(title="Total tested"),
        yaxis2=dict(title="Positivity %", overlaying="y", side="right", showgrid=False),
    )
    fig = plotly_template(fig)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Volume shown as bars (left axis), positivity rate as a line (right axis) — two different "
        "units on one chart, not two scales of the same measure, so this stays a single readable view "
        "rather than a true dual-axis comparison."
    )

    st.markdown('<h3>Trend check</h3>', unsafe_allow_html=True)
    volume_spikes = detect_spikes(trend, "TOTAL_TESTED")
    positivity_spikes = detect_spikes(trend, "POSITIVITY_RATE_PCT")
    n_spikes = len(volume_spikes) + len(positivity_spikes)
    if n_spikes == 0:
        note("No month in the selected period stands out as a statistical outlier in either testing volume "
             "or positivity rate — the trend stays within its normal range throughout.")
    else:
        note(f"<b>{n_spikes} month(s)</b> in this period stand out as statistical outliers in volume or "
             f"positivity rate. Open <b>Spike Deep-Dive</b> in the sidebar for the day-by-day breakdown, "
             f"the ranked contributing factors, and a recommendation for each one.")

    st.markdown('<h3>At a glance across every section</h3>', unsafe_allow_html=True)

    row_a = st.columns(3)
    with row_a[0].container(height=320, border=True):
        st.markdown("**Burden & Positivity**")
        df = get_positivity_by("SEX", start=start_date, end=end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by sex", color=CATEGORICAL[0],
                   pct=True, height=210)
    with row_a[1].container(height=320, border=True):
        st.markdown("**Testing Performance**")
        df = get_strategy_yield(start_date, end_date).head(5)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Top strategies by yield", color=CATEGORICAL[1],
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
        bar_chart(risk, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by risk-assessment status",
                   color=CATEGORICAL[7], pct=True, height=170)
    with row_b[1].container(height=320, border=True):
        st.markdown("**Kit / Algorithm QA**")
        alg = get_algorithm_flags(start_date, end_date).iloc[0]
        sub1, sub2 = st.columns(2)
        kpi_tile(sub1, "Expired kits used", int(alg.EXPIRED_KIT_USED),
                  color=status_color(alg.EXPIRED_KIT_USED, 0, 0, higher_is_better=False))
        kpi_tile(sub2, "Invalid algorithms", int(alg.INVALID_POSITIVE_ALGORITHM) + int(alg.INVALID_NEGATIVE_ALGORITHM),
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

# ---- Testing & Care Cascade ----
elif section == "Testing & Care Cascade":
    st.subheader("Who's being tested, and how positivity varies by group")
    left, right = st.columns(2)
    with left:
        df = get_positivity_by("SEX", start=start_date, end=end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by sex", color=CATEGORICAL[0], pct=True)
        note(insight_categorical(df, "sex"))
    with right:
        df = get_positivity_by("SETTING", start=start_date, end=end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by setting", color=CATEGORICAL[2], pct=True)
        note(insight_categorical(df, "setting", small_n=100))

    df = get_positivity_by("AGE_BAND", order_by="total_tested", start=start_date, end=end_date)
    bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by age band", color=CATEGORICAL[0], pct=True)
    note(insight_categorical(df, "age band"))

    df = get_positivity_by("MARITAL_STATUS_LABEL", order_by="total_tested", start=start_date, end=end_date)
    bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by marital status", color=CATEGORICAL[0],
               horizontal=True, pct=True)
    note(insight_categorical(df, "marital status"))

    st.markdown("---")
    st.subheader("Where testing happens, and how effective each route is")
    df = get_strategy_yield(start_date, end_date)
    bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Yield by testing strategy", color=CATEGORICAL[0],
               horizontal=True, pct=True)
    note(insight_categorical(df, "testing strategy", unit="Yield"))

    df = get_entrypoint_yield(start_date, end_date)
    bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Yield by entry point", color=CATEGORICAL[0],
               horizontal=True, pct=True)
    note(insight_categorical(df, "entry point", unit="Yield"))

    df = get_retest_interval(start_date, end_date)
    bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by time since last test",
               color=CATEGORICAL[0], pct=True)
    note(
        "The 'No prior test / not recorded' bucket mixes true first-time testers with repeat testers whose "
        "retest interval simply wasn't captured, so it overstates first-time volume — read it as an upper "
        "bound, not an exact count. " + insight_categorical(df, "retest interval")
    )

    st.markdown("---")
    st.subheader("Care cascade — from a positive result to linkage")
    kpis = get_kpis(start_date, end_date).iloc[0]
    c1, c2, c3 = st.columns(3)
    kpi_tile(c1, "Positives linked to care", f"{int(kpis.POSITIVES_LINKED)} / {int(kpis.TOTAL_POSITIVE)}",
              color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75))
    kpi_tile(c2, "Linkage rate", f"{kpis.LINKAGE_RATE_PCT:.1f}%",
              color=status_color(kpis.LINKAGE_RATE_PCT, 90, 75))
    kpi_tile(c3, "Result-given rate", f"{kpis.RESULT_GIVEN_RATE_PCT:.1f}%",
              color=status_color(kpis.RESULT_GIVEN_RATE_PCT, 95, 90))
    note(insight_linkage(kpis))

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Unlinked positive clients — needs follow-up**")
    unlinked = get_unlinked_positives(start_date, end_date)
    table_view(unlinked, caption=f"{len(unlinked)} clients flagged as positive but not linked to care.")

# ---- Clinical & Data Quality ----
elif section == "Clinical & Data Quality":
    st.subheader("Clinical correlations")
    risk = get_risk_assessment(start_date, end_date)
    zero_positive_when_assessed = (
        risk.loc[risk["CATEGORY"] == True, "TOTAL_POSITIVE"].sum() == 0  # noqa: E712
        if True in risk["CATEGORY"].values else False
    )
    if zero_positive_when_assessed:
        st.markdown(
            """
            <div class="flag-box">
              <b>⚠️ Flagged for medical review — ASSESSEDFORHIVRISK</b>
              <p>Zero positive results among clients formally risk-assessed; every positive came from
              clients where this field is FALSE. This is the opposite of the expected relationship and
              should be reviewed with the clinical team before this field is used elsewhere.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    bar_chart(risk, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by risk-assessment status",
               color=CATEGORICAL[7], pct=True)
    if not zero_positive_when_assessed:
        note(insight_categorical(risk, "risk-assessment status"))

    left, right = st.columns(2)
    with left:
        df = get_tb_screening(start_date, end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by TB screening status",
                   color=CATEGORICAL[1], horizontal=True, pct=True)
        note(insight_categorical(df, "TB screening status"))
    with right:
        df = get_self_test(start_date, end_date)
        bar_chart(df, "CATEGORY", "POSITIVITY_RATE_PCT", "Positivity by self-test history",
                   color=CATEGORICAL[2], pct=True)
        note(insight_categorical(df, "self-test history"))

    couple_df = get_couple_testing(start_date, end_date)
    disc_df = get_couple_discordance(start_date, end_date)
    left, right = st.columns(2)
    with left:
        bar_chart(couple_df, "CATEGORY", "TOTAL_ENCOUNTERS", "Individual vs. couple testing", color=CATEGORICAL[4])
    with right:
        bar_chart(disc_df, "CATEGORY", "TOTAL_COUPLES", "Discordance among couples tested", color=CATEGORICAL[4])
    note(insight_couple(couple_df, disc_df))

    st.markdown("---")
    st.subheader("Kit and testing-algorithm QA")
    left, right = st.columns([2, 1])
    with left:
        df = get_kit_brand(start_date, end_date)
        bar_chart(df, "CATEGORY", "TOTAL_USED", "Test kit brand distribution", color=CATEGORICAL[0])
    with right:
        conf = get_confirmatory_completion(start_date, end_date).iloc[0]
        kpi_tile(right, "Positives with 2nd algorithm test",
                  f"{int(conf.HAS_TEST2_RESULT)} / {int(conf.TOTAL_POSITIVE)}")
        kpi_tile(right, "Positives with 3rd algorithm test",
                  f"{int(conf.HAS_TEST3_RESULT)} / {int(conf.TOTAL_POSITIVE)}")
    note(insight_confirmatory(conf))

    alg = get_algorithm_flags(start_date, end_date).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    kpi_tile(c1, "Expired kit used", int(alg.EXPIRED_KIT_USED),
              color=status_color(alg.EXPIRED_KIT_USED, 0, 0, higher_is_better=False))
    kpi_tile(c2, "Invalid positive algorithm", int(alg.INVALID_POSITIVE_ALGORITHM),
              color=status_color(alg.INVALID_POSITIVE_ALGORITHM, 0, 2, higher_is_better=False))
    kpi_tile(c3, "Invalid negative algorithm", int(alg.INVALID_NEGATIVE_ALGORITHM),
              color=status_color(alg.INVALID_NEGATIVE_ALGORITHM, 0, 2, higher_is_better=False))
    kpi_tile(c4, "Discordant confirmatory", int(alg.DISCORDANT_CONFIRMATORY),
              color=status_color(alg.DISCORDANT_CONFIRMATORY, 0, 2, higher_is_better=False))
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

    trend = get_flag_monthly_trend(start_date, end_date)
    fig = go.Figure()
    fig.add_scatter(x=trend["VISIT_MONTH"], y=trend["FLAG_RATE_PCT"], mode="lines+markers",
                     marker_color=CATEGORICAL[7], line=dict(width=2))
    fig.update_layout(title="Monthly 'any flag raised' rate", height=380)
    fig = plotly_template(fig, y_title="%")
    st.plotly_chart(fig, width="stretch")
    note(insight_trend(trend, "FLAG_RATE_PCT", "the data-quality flag rate"))

# ---- Provider Performance ----
elif section == "Provider Performance":
    prov = get_provider_performance(start_date, end_date)
    prov_display = prov.copy()
    prov_display["PROVIDER_ID"] = prov_display["PROVIDER_ID"].str[:10] + "…"
    table_view(
        prov_display,
        caption="Provider IDs are hashed — map back to real facility names internally if you keep that key. "
                "Sort by FLAG_RATE_PCT or LINKAGE_RATE_PCT to spot outlier sites.",
    )
    note(insight_provider_outliers(prov))

    fig = go.Figure()
    fig.add_scatter(
        x=prov["TOTAL_TESTED"], y=prov["POSITIVITY_RATE_PCT"], mode="markers",
        marker=dict(
            size=(prov["FLAG_RATE_PCT"].fillna(0) + 4) * 2.5,
            color=CATEGORICAL[0],
            line=dict(width=1, color=INK["primary"]),
        ),
        text=prov_display["PROVIDER_ID"],
        hovertemplate="Provider %{text}<br>Tested: %{x}<br>Positivity: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(title="Provider volume vs. positivity (bubble size = flag rate)", height=420)
    fig = plotly_template(fig, y_title="Positivity %")
    fig.update_xaxes(title="Total tested")
    st.plotly_chart(fig, width="stretch")

# ---- Spike Deep-Dive ----
elif section == "Spike Deep-Dive":
    st.subheader("What's behind the spikes — and what to do about them")
    st.caption(
        "Each anomalous month below gets the same three-part treatment: a day-by-day chart to check whether "
        "it was a short event or a sustained shift, a ranked chart of every tracked factor's share of that "
        "month against its usual share, and the data-quality flag rate for context — then a recommendation "
        "that draws on all three."
    )

    trend = get_monthly_trend(start_date, end_date)
    volume_spikes = detect_spikes(trend, "TOTAL_TESTED")
    positivity_spikes = detect_spikes(trend, "POSITIVITY_RATE_PCT")

    if not volume_spikes and not positivity_spikes:
        note("No month in the selected period stands out as a statistical outlier in either testing volume "
             "or positivity rate — the trend stays within its normal range throughout, so there's nothing "
             "here that needs a deeper look.")
    else:
        kpis_all = get_kpis(start_date, end_date).iloc[0]
        days_in_period = max((end_date - start_date).days + 1, 1)
        avg_daily_tested = float(kpis_all.TOTAL_TESTED) / days_in_period
        avg_positivity = float(kpis_all.POSITIVITY_RATE_PCT) if not pd.isna(kpis_all.POSITIVITY_RATE_PCT) else None

        dimension_mixes = [
            ("Entry point", get_monthly_entrypoint_mix(start_date, end_date), 0),
            ("Testing strategy", get_monthly_strategy_mix(start_date, end_date), 0),
            ("Setting", get_monthly_dimension_mix("SETTING", start_date, end_date), 0),
            ("Age band", get_monthly_dimension_mix("AGE_BAND", start_date, end_date), 0),
            ("Sex", get_monthly_dimension_mix("SEX", start_date, end_date), 0),
            ("Retest status", get_monthly_retest_mix(start_date, end_date), 0),
            ("Kit brand", get_monthly_dimension_mix("HIVTEST1_NAME", start_date, end_date), 20),
            ("Provider", get_monthly_dimension_mix("PROVIDER_ID", start_date, end_date), 20),
        ]

        flag_trend = get_flag_monthly_trend(start_date, end_date)
        avg_flag_rate = float(flag_trend["FLAG_RATE_PCT"].mean()) if not flag_trend.empty else 0.0

        if volume_spikes:
            st.markdown('<h3>Volume spikes</h3>', unsafe_allow_html=True)
            for spike in volume_spikes:
                render_spike_block(
                    spike, metric_label="volume", share_col="TOTAL_TESTED", share_of="testing volume",
                    chart_col="TOTAL_TESTED", chart_y_title="Tests conducted",
                    concentration_col="TOTAL_TESTED", avg_reference=avg_daily_tested,
                    dimension_mixes=dimension_mixes, flag_trend=flag_trend, avg_flag_rate=avg_flag_rate,
                    color=CATEGORICAL[0],
                )

        if positivity_spikes:
            st.markdown('<h3>Positivity-rate spikes</h3>', unsafe_allow_html=True)
            for spike in positivity_spikes:
                render_spike_block(
                    spike, metric_label="positivity", share_col="TOTAL_POSITIVE", share_of="positive results",
                    chart_col="POSITIVITY_RATE_PCT", chart_y_title="Positivity %",
                    concentration_col="TOTAL_POSITIVE", avg_reference=avg_positivity,
                    dimension_mixes=dimension_mixes, flag_trend=flag_trend, avg_flag_rate=avg_flag_rate,
                    color=CATEGORICAL[7],
                )