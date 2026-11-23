"""Shared presentation: theme, KPI cards, insight cards, formatting and chart defaults."""
from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as C

CSS = f"""
<style>
:root {{ --primary:{C.PRIMARY}; --ink:{C.INK}; --muted:{C.MUTED}; --border:{C.BORDER}; }}
/* Streamlit's top bar overlays the page: keep it transparent, hide the developer Deploy button, and start the
   page below it so the header is never hidden. */
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 2rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.ksh-header {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  border-bottom:1px solid var(--border); padding-bottom:.7rem; margin-bottom:.8rem; }}
.ksh-brand {{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; min-width:0; }}
@media (max-width: 760px) {{ .ksh-rule {{ display:none; }} .ksh-title {{ font-size:1.2rem; }} .ksh-logo {{ height:40px; }} }}
.ksh-logo {{ height:48px; width:auto; background:#fff; border-radius:8px; padding:4px 8px; }}
.ksh-rule {{ width:1px; height:38px; background:var(--border); }}
.ksh-title {{ font-size:1.45rem; font-weight:700; line-height:1.2; }}
.ksh-sub {{ opacity:.7; font-size:.86rem; }}
.ksh-asof {{ background:#ECFDF5; color:var(--primary); border:1px solid #A7F3D0; border-radius:999px;
  padding:.2rem .7rem; font-size:.8rem; font-weight:600; white-space:nowrap; }}
.ksh-chips {{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.1rem 0 .6rem 0; }}
.ksh-chip {{ background:#F3F4F6; border:1px solid var(--border); color:#374151; border-radius:999px;
  padding:.1rem .6rem; font-size:.75rem; }}
.kpi {{ border:1px solid var(--border); border-radius:12px; padding:.8rem .95rem; background:#fff; height:100%; }}
.kpi-label {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; font-weight:600; hyphens:auto; overflow-wrap:normal; word-break:normal; }}
.kpi-value {{ color:var(--ink); font-size:clamp(1.15rem, 1.9vw, 1.65rem); font-weight:700; margin-top:.15rem; line-height:1.15; }}
.kpi-note {{ color:var(--muted); font-size:.78rem; margin-top:.2rem; display:-webkit-box; -webkit-line-clamp:2;
  -webkit-box-orient:vertical; overflow:hidden; }}
.kpi[title] .kpi-label::after {{ content:" ⓘ"; opacity:.6; }}
.kpi[title] {{ cursor:help; }}
.kpi-bad {{ color:{C.RED}; }} .kpi-good {{ color:{C.PRIMARY}; }} .kpi-warn {{ color:{C.AMBER}; }}
.kpi.accent-critical {{ border-left:4px solid {C.RED}; }} .kpi.accent-warning {{ border-left:4px solid {C.AMBER}; }}
.kpi.accent-good {{ border-left:4px solid {C.PRIMARY_LIGHT}; }}
.insight {{ border:1px solid var(--border); border-radius:10px; padding:.65rem .8rem; margin-bottom:.5rem; background:#fff; }}
.insight.critical {{ border-left:4px solid {C.RED}; background:#FEF2F2; }}
.insight.warning {{ border-left:4px solid {C.AMBER}; background:#FFFBEB; }}
.insight.info {{ border-left:4px solid #60A5FA; }}
.insight.positive {{ border-left:4px solid {C.PRIMARY_LIGHT}; background:#F0FDF4; }}
.insight-title {{ font-weight:650; color:var(--ink); font-size:.92rem; }}
.insight-detail {{ color:#374151; font-size:.84rem; margin-top:.15rem; }}
.insight-tab {{ color:var(--muted); font-size:.72rem; margin-top:.25rem; }}
.section {{ font-size:1.02rem; font-weight:700; margin:1.1rem 0 .1rem 0; }}
.section-sub {{ opacity:.7; font-size:.82rem; margin-bottom:.4rem; }}
.ai-box {{ border:1px solid #C7D2FE; background:#F5F7FF; border-radius:12px; padding:.8rem 1rem; }}
.ai-tag {{ font-size:.7rem; font-weight:700; color:#4338CA; letter-spacing:.05em; text-transform:uppercase; }}
.ai-text {{ color:#1F2937; font-size:.98rem; line-height:1.5; margin-top:.25rem; }}
.insight {{ min-height: 6.6rem; }}
div[data-testid="stSegmentedControl"] button p {{ font-size:.86rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ── formatting ───────────────────────────────────────────────────────────────
def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def kes(x, short: bool = True) -> str:
    if _bad(x):
        return "—"
    if short and abs(x) >= 999_500:          # 999,600 must read KES 1.0M, not "KES 1,000K"
        return f"KES {x / 1e6:,.1f}M"
    if short and abs(x) >= 1e3:
        return f"KES {x / 1e3:,.0f}K"
    return f"KES {x:,.0f}"


def num(x, dp: int = 0) -> str:
    return "—" if _bad(x) else f"{x:,.{dp}f}"


def pct(x, dp: int = 0, signed: bool = False) -> str:
    if _bad(x):
        return "—"
    return f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


# ── components ───────────────────────────────────────────────────────────────
def kpi(label: str, value: str, note: str = "", tone: str = "", accent: str = "", tip: str = "") -> str:
    """KPI card. Keep `note` to one short line; put definitions and caveats in `tip` (shown on hover)."""
    tone_cls = {"bad": "kpi-bad", "good": "kpi-good", "warn": "kpi-warn"}.get(tone, "")
    title = f' title="{esc(tip)}"' if tip else ""
    return (f'<div class="kpi {("accent-" + accent) if accent else ""}"{title}><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value {tone_cls}">{esc(value)}</div>'
            f'<div class="kpi-note">{esc(note)}</div></div>')


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def insight_card(i: dict) -> str:
    return (f'<div class="insight {esc(i["severity"])}"><div class="insight-title">{esc(i["title"])}</div>'
            f'<div class="insight-detail">{esc(i["detail"])}</div></div>')


def open_tab(name: str) -> None:
    """Button callback: switch the top navigation to another tab (runs before the rerun, so filters are kept)."""
    st.session_state["nav"] = name


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section">{esc(title)}</div>' +
                (f'<div class="section-sub">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def chips(items: list[str]) -> None:
    if items:
        st.markdown('<div class="ksh-chips">' + "".join(f'<span class="ksh-chip">{esc(c)}</span>' for c in items)
                    + "</div>", unsafe_allow_html=True)


def empty(msg: str = "Nothing matches the current filters.") -> None:
    st.info(msg, icon=":material/filter_alt_off:")


def class_label(df: pd.DataFrame) -> pd.Series:
    """Therapeutic class for grouping; items with none (e.g. medical supplies) fall back to their product class."""
    tc = df["therapeutic_class"].replace("Unclassified", pd.NA)
    return tc.fillna(df["product_class"].fillna("Other") + " (no therapeutic class)")


def download(df: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=f"{name}.csv",
                       mime="text/csv", icon=":material/download:")


# ── charts ───────────────────────────────────────────────────────────────────
def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(size=12, color=C.INK),
                      showlegend=legend, legend=dict(orientation="h", y=1.08, x=0, title=None),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=C.BORDER)
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
