"""AI features, kept narrow and grounded:

1. briefing_summary  — rewrites the computed insights into a short executive briefing. Numbers come only
                        from the facts passed in; a deterministic summary is used when no model is available.
2. ask               — natural-language questions answered with ONE read-only SELECT over the V3 inventory
                        marts/dims/daily fact. The SQL is validated (allow-list of objects, single SELECT, no
                        DDL/DML, row cap) before it runs, and is shown to the user.

Provider chain is the dashboard's existing one: Groq → xAI → Anthropic (keys from st.secrets or env).
Only aggregate inventory data is ever sent to a model — the allowed objects contain no patient columns.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd
import streamlit as st

from . import db

# ── provider ─────────────────────────────────────────────────────────────────
# Preference lists: the first model the key can actually use wins (providers retire models over time).
PREFERENCES = {
    "groq": {"fast": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.1-8b-instant"],
             "smart": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]},
    "grok": {"fast": ["grok-3-mini"], "smart": ["grok-3-mini"]},
    "claude": {"fast": ["claude-haiku-5-5"], "smart": ["claude-sonnet-5-5"]},
}
_OVERRIDE = {("groq", "fast"): "KSH_AI_GROQ_FAST", ("groq", "smart"): "KSH_AI_GROQ_SMART"}


@st.cache_resource(ttl=86400, show_spinner=False)
def _available_models(p: str) -> tuple:
    if p not in ("groq", "grok"):
        return ()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_secret("GROQ_API_KEY" if p == "groq" else "XAI_API_KEY"),
                        base_url="https://api.groq.com/openai/v1" if p == "groq" else "https://api.x.ai/v1", timeout=20)
        return tuple(m.id for m in client.models.list().data)
    except Exception:  # noqa: BLE001
        return ()


def model_for(p: str, tier: str) -> str:
    env = _OVERRIDE.get((p, tier))
    if env and os.getenv(env):
        return os.environ[env]
    prefs = PREFERENCES[p][tier]
    avail = _available_models(p)
    return next((m for m in prefs if m in avail), prefs[0])


def _secret(key: str) -> str | None:
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:  # noqa: BLE001  (no secrets.toml)
        pass
    db._load_env()
    return os.getenv(key)


def provider() -> str:
    for name, key in (("groq", "GROQ_API_KEY"), ("grok", "XAI_API_KEY"), ("claude", "ANTHROPIC_API_KEY")):
        if _secret(key):
            return name
    return "none"


def available() -> bool:
    return provider() != "none"


def complete(system: str, user: str, max_tokens: int = 400, tier: str = "fast") -> str | None:
    p = provider()
    try:
        if p in ("groq", "grok"):
            from openai import OpenAI
            client = OpenAI(api_key=_secret("GROQ_API_KEY" if p == "groq" else "XAI_API_KEY"),
                            base_url="https://api.groq.com/openai/v1" if p == "groq" else "https://api.x.ai/v1",
                            timeout=45)
            model = model_for(p, tier)
            kwargs = dict(model=model, temperature=0.1,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if "gpt-oss" in model:   # reasoning models: reasoning tokens share the budget
                kwargs.update(max_tokens=max_tokens + 1500, reasoning_effort="low")
            else:
                kwargs.update(max_tokens=max_tokens)
            r = client.chat.completions.create(**kwargs)
            text = (r.choices[0].message.content or "").strip()
            return text or None
        if p == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"), timeout=45)
            msg = client.messages.create(model=model_for(p, tier), max_tokens=max_tokens, system=system,
                                         messages=[{"role": "user", "content": user}])
            return msg.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001
        st.session_state["ai_last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return None


# ── 1. briefing ──────────────────────────────────────────────────────────────
_BRIEF_SYSTEM = (
    "You are the inventory analyst for Kisumu Specialists Hospital. Write a 2 to 3 sentence summary for the "
    "hospital's managers: the overall stock situation, then the single most important action today. The detailed "
    "findings are shown right below as separate cards, so do NOT list them one by one. Use ONLY the facts provided; "
    "never invent or recompute numbers; copy at most 3 figures exactly as given. Plain sentences: no bullets, no "
    "headings, no preamble.")


def fallback_summary(facts: dict) -> str:
    """Plain summary used when no AI provider is configured (same shape as the AI one: 2 sentences)."""
    k, ins = facts.get("kpis", {}), facts.get("insights", [])
    first = (f"{k.get('shortages_to_act_on', 0):,} store items need ordering now and {k.get('out_of_stock_now', 0):,} "
             f"are out of stock.")
    return first + (f" Most urgent: {ins[0]['title'][0].lower() + ins[0]['title'][1:]}." if ins else
                    " No other exceptions at the current filters.")


@st.cache_data(ttl=3600, show_spinner=False)
def briefing_summary(facts_json: str) -> tuple[str, bool]:
    """Returns (markdown, is_ai). Cached per set of facts so reruns cost nothing."""
    facts = json.loads(facts_json)
    if available():
        text = complete(_BRIEF_SYSTEM, "Facts (JSON):\n" + facts_json, max_tokens=220, tier="smart")
        if text:
            return text, True
    return fallback_summary(facts), False


# ── 2. ask the data ──────────────────────────────────────────────────────────
R = "HOSPITALS.REPORTING"
SCHEMA = {
    "RPT_KSH_V3_INV_STOCK_STATUS": (
        "Current stock per active store x product (as_of_date = data date). Columns: as_of_date, store_id, store_name, "
        "store_type, product_id, product_name, product_class, unit_std, therapeutic_class, therapeutic_subclass, "
        "clinical_tier (CRITICAL/HIGH/STANDARD), qty_on_hand, unit_cost, stock_value (KES), consumption_90d, "
        "dispensing_90d, ward_use_90d, adc_90d (avg daily consumption), days_of_stock, stock_status "
        "(Negative/Stocked out/Critical/Low/Adequate/No recent demand), last_consumed_date, days_since_last_consumed, "
        "priority_action (ORDER NOW/ORDER THIS WEEK/MONITOR/CHECK STOCK RECORD — the last = negative balance with no use), priority_score. IMPORTANT: days_of_stock is NULL when "
        "qty_on_hand <= 0 or there is no demand, so for 'out of stock / running low / below N days' questions filter on "
        "stock_status IN ('Negative','Stocked out','Critical'[, 'Low']) rather than days_of_stock alone, and only "
        "count items with adc_90d > 0 as having demand."),
    "RPT_KSH_V3_INV_STOCKOUTS": (
        "One row per stockout episode per store x product. Columns: store_id, store_name, store_type, product_id, "
        "product_name, product_class, therapeutic_class, clinical_tier, episode_no, episodes_for_item, "
        "stockout_start_date, restock_date, is_ongoing, days_out, restocked_by (Receipt / Transfer in / Stock count / Manual adjustment / System adjustment / Ledger jump (no movement)), started_on_system_adjustment. For 'out of stock now' use is_ongoing AND NOT started_on_system_adjustment."),
    "RPT_KSH_V3_INV_PRODUCT_PERFORMANCE": (
        "One row per product (all active stores). Columns: product_id, product_name, product_class, therapeutic_class, "
        "therapeutic_subclass, clinical_tier, qty_on_hand, stock_value, unit_cost, consumption_30d, consumption_90d, "
        "consumption_365d, dispensing_365d, ward_use_365d, consumption_value_365d, last_consumed_date, "
        "days_since_last_consumed, cv_30d_periods_12, turnover_365d, dispensed_value_90d (KES), rx_lines_365d, "
        "rx_dispensed_lines_365d, rx_fill_rate_365d, abc_class (A/B/C within product_class), movement_status "
        "(Active/Slow/Dead/No stock), months_to_clear, change_vs_prev_30d"),
    "RPT_KSH_V3_INV_CONTROL": (
        "Monthly control metrics, long format. Columns: month, store_id, store_name, store_type, metric, n_events, "
        "qty, value_kes. metric values: 'Losses (negative adjustments + write-offs)', 'Stock-count gains', "
        "'Stock-count losses', 'Stock-count net loss before re-sync netting' (do not add this to the others), 'Manual adjustments (net, excl. re-sync reversals)' (staff edits other than undoing a same-month system re-sync), 'Manual adjustments undoing a system re-sync (net)' (bookkeeping clean-up, not a loss), 'Transfers in', 'Transfers out', "
        "'Dispenses from negative stock', 'Prescriptions dispensed', 'Prescriptions cancelled', "
        "'Prescriptions stopped', 'Prescriptions not dispensed', 'Unfilled prescriptions: item out of stock' (a subset of not dispensed — never add it to the others). Fill rate = dispensed / (dispensed + not dispensed); cancelled and stopped are excluded. month is a DATE (first day of the month); the "
        "latest month is partial. For 'last N months' use complete months: month >= DATEADD(month, -N, (SELECT "
        "MAX(month) FROM HOSPITALS.REPORTING.RPT_KSH_V3_INV_CONTROL)) AND month < (SELECT MAX(month) FROM "
        "HOSPITALS.REPORTING.RPT_KSH_V3_INV_CONTROL). Net shrinkage value = SUM(value_kes) of 'Stock-count losses' "
        "minus SUM(value_kes) of 'Stock-count gains' (use SUM(IFF(metric=...))). store_name is NULL for "
        "facility-wide rows."),
    "FACT_KSH_V3_INV_DAILY": (
        "Daily movements per store x product. Columns: movement_date, store_id, product_id, dispensed_qty, "
        "patient_return_qty, ward_use_qty, consumable_return_qty, net_dispensing_qty, net_consumable_qty, "
        "net_consumption_qty, receipt_qty, transfer_in_qty, transfer_out_qty, loss_qty, stock_count_adj_qty, "
        "manual_adj_qty, day_opening, day_closing. Join DIM_KSH_V3_INV_PRODUCT / DIM_KSH_V3_INV_STORE for names."),
    "DIM_KSH_V3_INV_PRODUCT": (
        "Products. Columns: product_id, product_name, product_class, category_name, unit_std, therapeutic_class, "
        "therapeutic_subclass, clinical_tier, inn_name"),
    "DIM_KSH_V3_INV_STORE": (
        "Stores. Columns: store_id, store_name, store_type, is_active_store, facility_name"),
}
ALLOWED = {f"{R}.{t}" for t in SCHEMA}
_FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE",
              "CALL", "COPY", "PUT", "GET", "USE", "EXECUTE", "BEGIN", "COMMIT", "ROLLBACK", "UNDROP", "SHOW",
              "DESCRIBE", "UNSET", "REMOVE", "LIST"}
_BLOCKED_IDS = {"RESULT_SCAN", "IDENTIFIER", "TABLE", "GENERATOR", "INFORMATION_SCHEMA", "ACCOUNT_USAGE",
                "SNOWFLAKE", "READER_ACCOUNT_USAGE", "ORGANIZATION_USAGE", "INFER_SCHEMA", "VALIDATE"}
_CLAUSE_END = {"WHERE", "GROUP", "HAVING", "QUALIFY", "ORDER", "LIMIT", "UNION", "EXCEPT", "INTERSECT", "MINUS",
               "WINDOW", "FETCH", "OFFSET", "SELECT"}
_NOT_ALIAS = _CLAUSE_END | {"ON", "USING", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL",
                            "LATERAL", "AS", "AT", "BEFORE", "CHANGES", "SAMPLE", "TABLESAMPLE", "PIVOT", "UNPIVOT",
                            "MATCH_RECOGNIZE", "ASOF"}
_FROM_FUNCS = {"EXTRACT", "TRIM", "SUBSTRING", "POSITION", "OVERLAY"}   # their "FROM" is not a table source
_TOKEN = re.compile(r"""
    (?P<ws>\s+) | (?P<cmt>--[^\n]*|//[^\n]*|/\*.*?\*/) | (?P<qid>"(?:[^"]|"")+") | (?P<str>'(?:[^'\\]|\\.|'')*')
  | (?P<dollar>\$\$) | (?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?) | (?P<id>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<op>::|=>|<=|>=|<>|!=|\|\||[(),.;*+\-/%<>=:\[\]{}~^&|!?]) | (?P<bad>.)""", re.X | re.S)


def _tokens(sql: str) -> list[tuple[str, str]]:
    out = []
    for m in _TOKEN.finditer(sql):
        kind = m.lastgroup
        if kind in ("ws", "cmt"):
            continue
        out.append((kind, m.group()))
    return out


def _ident(tok: tuple[str, str]) -> str:
    return tok[1][1:-1].replace('""', '"').upper() if tok[0] == "qid" else tok[1].upper()


def validate_sql(sql: str) -> tuple[bool, str]:
    """Token-level allow-list: every table source (FROM / JOIN / comma join / lateral) must be one of the 7
    REPORTING objects or a CTE. Rejects stages (@), positional $ columns, dollar strings, table functions other
    than FLATTEN, parenthesised table names, write/DDL keywords and system schemas."""
    s = sql.strip().rstrip(";").strip()
    toks = _tokens(s)
    if not toks:
        return False, "Empty query."
    bad = [t for k, t in toks if k in ("bad", "dollar")]
    if bad:
        return False, f"The query contains a character that isn't allowed ({bad[0]!r})."
    if any(t == ";" for _, t in toks):
        return False, "Only one statement is allowed."
    if toks[0][0] != "id" or toks[0][1].upper() not in ("SELECT", "WITH"):
        return False, "Only SELECT queries are allowed."
    ids = [t.upper() for k, t in toks if k == "id"]
    if _FORBIDDEN & set(ids):
        return False, "The query contains a non-read keyword."
    names = {_ident(t) for t in toks if t[0] in ("id", "qid")}
    if (_BLOCKED_IDS & names) or any(n.startswith("SYSTEM$") for n in names):
        return False, "The query uses a blocked function or schema."
    ctes = {_ident(toks[i]) for i in range(len(toks) - 2)
            if toks[i][0] in ("id", "qid") and toks[i + 1][1].upper() == "AS" and toks[i + 2][1] == "("}

    def source(j: int) -> tuple[int, str | None]:
        """Validate one table source starting at token j; return (next index, error)."""
        if j < len(toks) and toks[j][1].upper() == "LATERAL":
            j += 1
        if j >= len(toks):
            return j, "Incomplete FROM clause."
        if toks[j][1] == "(":
            nxt = toks[j + 1][1].upper() if j + 1 < len(toks) else ""
            # subquery: hand back the "(" so the main loop tracks depth and checks the inner FROMs
            return (j, None) if nxt in ("SELECT", "WITH") else (j, "Parenthesised table names aren't allowed.")
        if toks[j][0] not in ("id", "qid"):
            return j, "Only the V3 inventory views may be queried."
        parts = [_ident(toks[j])]
        j += 1
        while j + 1 < len(toks) and toks[j][1] == "." and toks[j + 1][0] in ("id", "qid"):
            parts.append(_ident(toks[j + 1]))
            j += 2
        if j < len(toks) and toks[j][1] == "(":                       # table function
            return (j, None) if parts == ["FLATTEN"] else (j, "Table functions other than FLATTEN aren't allowed.")
        name = ".".join(parts)
        ok = (len(parts) == 1 and parts[0] in ctes) or \
             (len(parts) == 2 and f"HOSPITALS.{name}" in ALLOWED) or (len(parts) == 3 and name in ALLOWED)
        if not ok:
            return j, f"Only the V3 inventory views may be queried, fully qualified (found {name})."
        if j < len(toks) and toks[j][1].upper() == "AS":
            j += 1
        if j < len(toks) and toks[j][0] in ("id", "qid") and toks[j][1].upper() not in _NOT_ALIAS:
            j += 1
        return j, None

    depth, parens, from_depths = 0, [], []      # parens: function name (or "") per open paren
    i = 0
    while i < len(toks):
        kind, t = toks[i]
        up = t.upper()
        if t == "(":
            parens.append(toks[i - 1][1].upper() if i and toks[i - 1][0] == "id" else "")
            depth += 1
        elif t == ")":
            depth -= 1
            parens and parens.pop()
            from_depths = [d for d in from_depths if d <= depth]
        elif kind == "id" and up in _CLAUSE_END:
            from_depths = [d for d in from_depths if d != depth]
        if kind == "id" and up in ("FROM", "JOIN") and not (up == "FROM" and parens and parens[-1] in _FROM_FUNCS):
            if up == "FROM":
                from_depths.append(depth)
            i, err = source(i + 1)
            if err:
                return False, err
            continue
        if t == "," and depth in from_depths:                          # comma join inside a FROM clause
            i, err = source(i + 1)
            if err:
                return False, err
            continue
        i += 1
    return True, s


_SQL_SYSTEM = (
    "You write ONE Snowflake SQL SELECT statement that answers the user's question about hospital inventory. "
    "Use only these objects, always fully qualified as HOSPITALS.REPORTING.<name>:\n"
    + "\n".join(f"- {k}: {v}" for k, v in SCHEMA.items()) +
    "\nStock bands in stock_status: 'Critical' = under 7 days left, 'Low' = 7 to 30 days, 'Stocked out' = zero, "
    "'Negative' = below zero (all three of Negative/Stocked out/Critical are 'under 7 days'). For questions about "
    "items under 7 days / running out / out of stock, filter ONLY on stock_status (and adc_90d > 0) and do NOT add "
    "any days_of_stock condition (it is NULL for out-of-stock rows and would wrongly drop them). "
    "When ordering by a value DESC always add NULLS LAST, and exclude NULL values for 'top by value' questions."
    "\nRules: read-only SELECT (CTEs allowed); no semicolons; product and store names are upper/mixed case — use "
    "ILIKE for name matching; usage windows (last N days of consumption/dispensing) end on the last COMPLETE day = MAX(movement_date) - 1 in FACT_KSH_V3_INV_DAILY (the final day is partial); 'today' for stock means the data's as-of date (MAX(as_of_date) in RPT_KSH_V3_INV_STOCK_STATUS "
    "or MAX(movement_date) in FACT_KSH_V3_INV_DAILY), never CURRENT_DATE; money is KES; prefer the RPT_ views; "
    "order results meaningfully and return at most 50 rows. Keep queries simple: no table functions (FLATTEN, "
    "SPLIT_TO_TABLE, GENERATOR, TABLE(...)), filter dates with DATEADD on the as-of date. "
    "Reply with the SQL only, no explanation, no code fence.")

_EXPLAIN_SYSTEM = (
    "You answer a hospital pharmacy manager's question from a query result, in 2–4 plain sentences. Lead with the "
    "direct answer, then name the most important items (by product and store) with their key figures, exactly as "
    "they appear in the result. Mention at most 5 figures, copied exactly — never retype long tables; for longer "
    "results summarise the pattern (largest, smallest, trend) and say the full detail is in the table below. Use "
    "only the result; do not speculate. If the result is empty, say plainly that nothing matched. No preamble.")


def ask(question: str) -> dict:
    """Returns {sql, ok, error, df, answer}. Never raises."""
    out = {"question": question, "sql": None, "ok": False, "error": None, "df": None, "answer": None}
    if not available():
        out["error"] = "No AI provider is configured for this dashboard."
        return out
    prompt, df = question, None
    for attempt in range(2):          # one self-correction: the model sees why its first query was rejected
        raw = complete(_SQL_SYSTEM, prompt, max_tokens=600, tier="smart")
        if not raw:
            out["error"] = "The AI service did not respond. " + st.session_state.get("ai_last_error", "")
            return out
        sql = re.sub(r"^```(?:sql)?|```$", "", raw.strip(), flags=re.I | re.M).strip()
        out["sql"] = sql
        ok, checked = validate_sql(sql)
        if ok:
            try:
                df = db.query(f"SELECT * FROM ({checked}) LIMIT 500")
                break
            except Exception as exc:  # noqa: BLE001
                checked = f"The generated query failed: {str(exc).splitlines()[0][:200]}"
        out["error"] = checked
        prompt = (f"{question}\n\nYour previous query was rejected: {checked}\nPrevious query:\n{sql}\n"
                  "Write a simpler corrected query that follows the rules.")
    if df is None:
        return out
    out.update(ok=True, df=df, error=None)
    # Round before the model sees the figures, so the answer quotes 1,476,188 rather than 1476187.8399999994.
    head = df.head(30).copy()
    for c in head.select_dtypes("number").columns:
        big = head[c].abs().max() >= 100
        head[c] = head[c].map(lambda v, b=big: "" if pd.isna(v) else (f"{v:,.0f}" if b else f"{v:,.2f}"))
    preview = head.to_csv(index=False)
    out["answer"] = complete(_EXPLAIN_SYSTEM, f"Question: {question}\nResult (CSV, up to 30 rows):\n{preview}",
                             max_tokens=300, tier="smart")
    return out
