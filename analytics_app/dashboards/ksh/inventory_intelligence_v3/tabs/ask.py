"""Ask the data — plain-English questions answered with a validated, read-only query over the V3 marts."""
from __future__ import annotations

import streamlit as st

from .. import ai, ui

EXAMPLES = [
    "Which critical-tier medicines have less than 7 days of stock, and in which store?",
    "Top 10 products by stock value that have not been used for 90 days",
    "Monthly net stock-count shrinkage value by store this year",
    "Which antimicrobials had the most stockout days in the last 6 months?",
]


def render(ctx: dict) -> None:
    ui.section("Ask the data", "Ask in plain English. Answers come from the inventory data (no patient data), "
                               "with the query shown so you can check it.")
    if not ai.available():
        st.warning("No AI provider is configured for this dashboard (GROQ_API_KEY, XAI_API_KEY or "
                   "ANTHROPIC_API_KEY), so questions can't be answered here.", icon=":material/key_off:")
        return
    hist = st.session_state.setdefault("ask_history", [])
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for c, q in zip(cols, EXAMPLES):
        if c.button(q, use_container_width=True):
            clicked = q
    q = st.chat_input("Ask about stock, stockouts, demand, losses…") or clicked
    if q:
        with st.spinner("Thinking…"):
            hist.insert(0, ai.ask(q))
    for res in hist[:5]:
        with st.chat_message("user"):
            st.write(res["question"])
        with st.chat_message("assistant"):
            if res["ok"]:
                if res["answer"]:
                    st.markdown(res["answer"])
                st.dataframe(res["df"], hide_index=True, use_container_width=True)
                ui.download(res["df"], "ksh_v3_answer")
            else:
                st.error(res["error"], icon=":material/error:")
            if res["sql"]:
                with st.expander("Query used"):
                    st.code(res["sql"], language="sql")
