"""
prompt_builder.py — Assembles the LLM message list for the KSH AI Chat Layer.

Inputs:  question, route_result (from router.py), notices, facility, notice_date, history
Outputs: OpenAI-format message list — ready to send to the API, no calls made here.

Message order:
  1. System prompt (role + rules + active alerts)
  2. LIVE METRIC DATA block    (use_snapshot=True)
  3. QUERY RESULT block        (use_sql=True, sql_executor returned data)
  4. INVESTIGATION CONTEXT block (context files, char-budget-capped)
  5. Conversation history (last N turns)
  6. User question
"""

import threading
from pathlib import Path

_CHAT = Path(__file__).resolve().parent        # chat/
_PRIVATE = _CHAT.parent                        # facility_utilization/

_CONTEXT_CHAR_BUDGET = 16_000
_MAX_HISTORY_TURNS   = 6

# ---------------------------------------------------------------------------
# Context content cache — stem -> full file text, mtime-invalidated.
# Uses rglob so investigations/ files are included (same scope as router.py).
# ---------------------------------------------------------------------------
_content_cache: dict = {}
_content_mtime: float = 0.0
_content_lock  = threading.Lock()


def _get_context_content() -> dict:
    global _content_cache, _content_mtime
    ctx_dir = _PRIVATE / "context"
    try:
        latest = max(
            (f.stat().st_mtime for f in ctx_dir.rglob("*.md")), default=0.0
        )
    except OSError:
        latest = 0.0
    if latest <= _content_mtime:
        return _content_cache
    with _content_lock:
        try:
            latest = max(
                (f.stat().st_mtime for f in ctx_dir.rglob("*.md")), default=0.0
            )
        except OSError:
            latest = 0.0
        if latest <= _content_mtime:
            return _content_cache
        content = {}
        for f in ctx_dir.rglob("*.md"):
            content[f.stem] = f.read_text(encoding="utf-8")
        _content_cache = content
        _content_mtime = latest
    return _content_cache


# ---------------------------------------------------------------------------
# Unit formatting
# ---------------------------------------------------------------------------
_UNIT_SUFFIX = {
    "pct":               "%",
    "count":             "",
    "days":              " days",
    "sessions":          " sessions",
    "months_elapsed":    " months",
    "pct_of_personal_avg": "%",
}


def _fmt(value, unit: str) -> str:
    if value is None:
        return "N/A"
    suffix = _UNIT_SUFFIX.get(unit, f" {unit}")
    return f"{round(value, 2)}{suffix}"


# ---------------------------------------------------------------------------
# Alert status (indicative — consecutive-month logic lives in the alerting
# engine; active alerts in the system prompt are authoritative)
# ---------------------------------------------------------------------------
def _alert_status(value, alerting: dict) -> str:
    if not alerting.get("enabled"):
        return "alerting disabled"
    direction = alerting.get("direction", "above")
    watch    = alerting.get("watch")
    critical = alerting.get("critical")
    if value is None:
        return "unknown"
    if direction == "above":
        if critical is not None and value >= critical:
            return "CRITICAL"
        if watch    is not None and value >= watch:
            return "WATCH"
    elif direction == "below":
        if critical is not None and value <= critical:
            return "CRITICAL"
        if watch    is not None and value <= watch:
            return "WATCH"
    return "OK"


# ---------------------------------------------------------------------------
# Snapshot block
# ---------------------------------------------------------------------------
def _format_snapshot_block(matches: list) -> str:
    """
    Format matched metrics into a structured block the LLM can cite verbatim.
    Alert status is indicative only — active alerts in the system prompt are
    authoritative (they apply the full consecutive-month / min-gate rules).
    """
    if not matches:
        return ""

    lines = [
        "LIVE METRIC DATA — cite these values only; "
        "do not calculate, extrapolate, or invent figures.\n"
        "Alert status below is indicative; active alerts listed in the "
        "system prompt are authoritative.\n"
    ]

    for m in matches:
        snap = m.get("snapshot")
        reg  = m["registry"]
        unit = reg.get("unit", "")

        if snap is None:
            lines.append(f"\n[{reg['label']}]\ndata_quality: NOT AVAILABLE\n")
            continue

        cv  = snap.get("current_value")
        t3  = snap.get("trailing_3mo_avg")
        p3  = snap.get("prior_3mo_avg")
        fetch_ok = snap.get("fetch_ok", True)
        warnings = snap.get("warnings", [])
        history  = snap.get("history", [])

        # Quality flag
        if not fetch_ok:
            quality = "WARNING — fetch failed; values may be stale"
        elif "stale_data" in warnings:
            quality = "WARNING — latest month may be incomplete"
        else:
            quality = "OK"

        # Threshold string
        alerting  = reg.get("alerting", {})
        direction = alerting.get("direction", "above")
        watch_v   = alerting.get("watch")
        crit_v    = alerting.get("critical")
        thresh_parts = []
        if watch_v   is not None:
            thresh_parts.append(f"WATCH {direction} {_fmt(watch_v, unit)}")
        if crit_v    is not None:
            thresh_parts.append(f"CRITICAL {direction} {_fmt(crit_v, unit)}")
        thresh_str = " | ".join(thresh_parts)

        # History (newest first, up to 6 months)
        hist_parts = []
        for h in history[:6]:
            month = h.get("month", "")[:7]
            val   = h.get("value")
            if val is not None:
                hist_parts.append(f"{month}: {_fmt(val, unit)}")
        hist_str = " | ".join(hist_parts) or "no history"

        # Entity-level values (doctor metrics)
        entities    = snap.get("entities", {})
        entity_lines = []
        for name, data in entities.items():
            ev = data.get("current_value")
            if ev is not None:
                entity_lines.append(f"  {name}: {_fmt(ev, unit)}")

        block = [f"\n[{reg['label']}]"]
        block.append(f"current_value:    {_fmt(cv, unit)}")
        block.append(f"trailing_3mo_avg: {_fmt(t3, unit)}")
        block.append(f"prior_3mo_avg:    {_fmt(p3, unit)}")
        block.append(
            f"alert_status:     {_alert_status(cv, alerting)}"
            + (f" (thresholds: {thresh_str})" if thresh_str else "")
        )
        block.append(f"history:          {hist_str}")
        if entity_lines:
            block.append("per_entity:")
            block.extend(entity_lines)
        block.append(f"data_quality:     {quality}")
        lines.append("\n".join(block) + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL result block
# ---------------------------------------------------------------------------
def _format_sql_result_block(sql_result: dict) -> str:
    """
    Format a SQLResult dict into a structured block for the LLM.
    Only called when confidence is "validated" or "partial".
    Returns empty string for not_required, failed, or None.
    """
    if not sql_result:
        return ""
    confidence = sql_result.get("confidence", "failed")
    if confidence not in ("validated", "partial"):
        return ""

    rows    = sql_result.get("returned_rows", 0)
    columns = sql_result.get("columns", [])
    data    = sql_result.get("data", [])

    if confidence == "partial" or rows == 0:
        return (
            "QUERY RESULT — direct database query returned no rows.\n"
            "This means no data matched the filters for this question. "
            "Do not invent values — state that the database query found no matching records."
        )

    # Render as a compact table (up to 20 rows for prompt budget)
    header = " | ".join(columns)
    sep    = " | ".join("-" * max(len(c), 4) for c in columns)
    row_lines = []
    for row in data[:20]:
        row_lines.append(" | ".join(str(row.get(c, "")) for c in columns))
    table = "\n".join([header, sep] + row_lines)
    if rows > 20:
        table += f"\n... ({rows - 20} more rows not shown)"

    return (
        f"QUERY RESULT — cite these database values directly; do not calculate or extrapolate.\n"
        f"Returned {rows} row(s).\n\n"
        f"{table}"
    )


# ---------------------------------------------------------------------------
# Context block
# ---------------------------------------------------------------------------
def _format_context_block(context_keys: list) -> str:
    """
    Load context file content for the given stems, concatenated up to
    _CONTEXT_CHAR_BUDGET chars. First match is always included even if large.
    Falls back to all context files when context_keys is empty.
    """
    content_map = _get_context_content()
    keys = context_keys if context_keys else list(content_map.keys())

    parts     = []
    char_used = 0
    for key in keys:
        text = content_map.get(key)
        if not text:
            continue
        if char_used > 0 and char_used + len(text) > _CONTEXT_CHAR_BUDGET:
            break
        parts.append(text)
        char_used += len(text)

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt (matches views.py exactly — Phase 2.3 removes it from there)
# Rule 1 updated to reference LIVE METRIC DATA block explicitly.
# ---------------------------------------------------------------------------
def _build_system_prompt(facility: str, notice_date: str, notices: list) -> str:
    notice_block = "\n".join(
        f"[{n['level']}] {n['title']}\n  Metric: {n['metric']}\n  Action: {n['action']}"
        for n in notices
    ) or "No active alerts."
    return (
        f"You are a trusted clinical operations analyst for {facility} hospital in Kisumu, Kenya."
        f" Today: {notice_date}.\n\n"
        f"Active alerts this week:\n{notice_block}\n\n"
        f"Rules you must follow without exception:\n"
        f"1. Only cite metric values (percentages, visit counts, thresholds) that appear VERBATIM"
        f" in the INVESTIGATION CONTEXT, LIVE METRIC DATA, or QUERY RESULT blocks. Never calculate,"
        f" extrapolate, or invent figures. When a question asks about something the investigation has not"
        f" confirmed, use Status: No Evidence Found and state only what the investigation DID"
        f" establish on the nearest related topic — do not fill gaps with plausible reasoning"
        f" or general knowledge.\n"
        f"2. Present findings as analytical outputs, not operational prescriptions. Do not"
        f" recommend hiring, redistribution, or specific management actions. State what the"
        f" data shows and who should review it.\n"
        f"3. Use doctor usernames exactly as written in context (e.g. eawando, jogutu)."
        f" Never swap roles.\n"
        f"4. NEVER tell the user to 'check the dashboard' for operational questions — answer"
        f" directly from context. Exception: for causal or investigation questions, apply Rule 14.\n"
        f"5. When answering 'why' questions about operational metrics (volume, LOS, readmissions,"
        f" workload), lead with the strongest data finding available in context.\n"
        f"6. Present confirmed investigation findings as facts: 'the data shows',"
        f" 'the investigation confirmed'.\n"
        f"7. Distinguish between investigation findings and unresolved details. When the analysis"
        f" has identified the mechanism driving a metric, present that mechanism as a completed"
        f" finding. Frame remaining unknowns as narrower operational questions, not evidence"
        f" that the investigation is incomplete.\n"
        f"8. Format EVERY response — first and all follow-up — using this exact structure."
        f" Never deviate.\n\n"
        f"Status: [WATCH / CRITICAL / OK / Attention Required / No Evidence Found]\n\n"
        f"[Section label — choose: Key finding / Answer / Likely impact / Known limitation]\n"
        f"[One sentence only. Plain text, no bullet. State the most important insight.]\n\n"
        f"[Section label — choose: Evidence / Known facts / Related observations / Impact]\n"
        f"• [Metric or fact — one line]\n"
        f"• [Second — one line, only if genuinely distinct; omit if not needed]\n"
        f"• [Third — only if essential]\n\n"
        f"Strict rules for this format:\n"
        f"- Status is always first. It reflects what the data says about the issue,"
        f" not whether the response is complete.\n"
        f"- The first section (Key finding / Answer) is always a single plain-text"
        f" sentence — no bullet.\n"
        f"- The second section (Evidence / Known facts / Related observations) is always"
        f" bullet points — no prose.\n"
        f"- Never repeat a fact that appeared in the first section inside the second section.\n"
        f"- Never mix evidence and conclusions in the same section.\n"
        f"- No paragraphs, no narrative, no reasoning shown, no root-cause analysis,"
        f" no trends, no recommendations.\n\n"
        f"End every response with:\nFOLLOW-UPS:\n"
        f"• [most valuable next question]\n"
        f"• [second most valuable next question]\n"
        f"9. For broad questions: summarize ONLY what the investigation context explicitly covers.\n"
        f"10. For follow-up questions: apply the same format as Rule 8. Do not repeat any metric"
        f" or finding already stated in a prior response in this conversation — each follow-up"
        f" adds only new information. Answer only what was asked.\n"
        f"11. Findings that rule out explanations are completed analytical outcomes — state"
        f" them explicitly.\n"
        f"12. If the question cannot be answered from context or active alerts, use this structure:\n"
        f"Status: No Evidence Found\n\n"
        f"What's available\n"
        f"• [Closest confirmed finding from the investigation on this topic]\n"
        f"• [Second related finding if available]\n"
        f"Do not say 'the investigation did not test', 'the data does not show', or"
        f" 'we never investigated'. Do not explain what is missing. State only what IS available."
        f" Do not attempt to answer from general medical knowledge. No Evidence Found applies"
        f" only when investigation data is genuinely absent — do not use it when a question"
        f" contains action words like 'redistribution', 'options', or 'workload'; if the data"
        f" exists in context, report it directly.\n"
        f"13. Do NOT mention specific doctor names, departure events, or historical personnel"
        f" changes unless the question explicitly asks about a named person or a departure."
        f" State structural and pattern findings without attributing them to individuals. This"
        f" applies even when an active alert contains a doctor's name — describe the"
        f" concentration pattern without naming the individual. Only state a name if the user"
        f" explicitly asks 'which doctor' or names someone themselves.\n"
        f"14. For questions about causal findings, investigation results, or structural 'why'"
        f" questions — why are private wards empty, why is dialysis idle, what did the"
        f" investigation find, what drives readmissions, what is the concentration risk, what"
        f" happens to renal patients — respond with exactly this structure and nothing else:\n\n"
        f"Status: See Dashboard\n\n"
        f"Key finding\n"
        f"This is covered in the Causal Intelligence section of the KSH dashboard, where the"
        f" full investigation chain is documented with supporting data.\n\n"
        f"Evidence\n"
        f"• This chat answers live operational questions — ward activity, staffing workload,"
        f" theatre completion, lab volume, readmission rates.\n"
        f"• For investigation findings and causal analysis, open the Causal Intelligence page"
        f" on the dashboard."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_messages(
    question:     str,
    route_result: dict,
    notices:      list,
    facility:     str,
    notice_date:  str,
    history:      list,
    sql_result:   dict = None,
) -> list[dict]:
    """
    Assemble the LLM message list from router output, SQL result, and context.

    Args:
        question      The user's current question.
        route_result  Output from router.route(question).
        notices       Active alerts list from current_notices_KSH.json.
        facility      Facility name string (e.g. "KSH").
        notice_date   Date string for the system prompt.
        history       Conversation history in OpenAI message format.
        sql_result    Optional SQLResult dict from sql_executor.execute().

    Returns:
        List of {"role": ..., "content": ...} dicts ready for the API.
    """
    msgs: list[dict] = []

    # 1. System prompt
    msgs.append({
        "role":    "system",
        "content": _build_system_prompt(facility, notice_date, notices),
    })

    # 2. Snapshot block (use_snapshot=True)
    if route_result.get("use_snapshot"):
        snap_block = _format_snapshot_block(route_result.get("matches", []))
        if snap_block:
            msgs.append({"role": "system", "content": snap_block})

    # 3. Query result block (use_sql=True, validated/partial only)
    if sql_result:
        sql_block = _format_sql_result_block(sql_result)
        if sql_block:
            msgs.append({"role": "system", "content": sql_block})

    # 4. Context block (both paths)
    ctx_block = _format_context_block(route_result.get("context_keys", []))
    if ctx_block:
        msgs.append({
            "role":    "system",
            "content": f"INVESTIGATION CONTEXT:\n{ctx_block}",
        })

    # 5. Conversation history
    msgs.extend(history[-_MAX_HISTORY_TURNS:])

    # 6. Current question
    msgs.append({"role": "user", "content": question})

    return msgs
