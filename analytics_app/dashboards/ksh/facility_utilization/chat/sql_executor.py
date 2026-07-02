"""
sql_executor.py — SQL generation and execution for the KSH AI Chat Layer.

Called from views.py when route_result["use_sql"] is True.

Flow:
  1. not_required fast-exit — skip SQL when snapshot already answers the question.
  2. Derive schema summary from schema_catalog.json.
  3. Build generation prompt from sql_prompt.md + schema + question.
  4. Call LLM (Groq) to generate SQL.
  5. Validate with sql_validator.py — one retry on validation failure.
  6. Execute on Snowflake read-only connection.
  7. Return SQLResult dict.
  8. Append full audit record to sql_requests.jsonl.

SQLResult schema:
  {
    "question":      str,
    "generated_sql": str | None,
    "returned_rows": int,
    "columns":       list[str],
    "data":          list[dict],
    "confidence":    "validated" | "partial" | "failed" | "not_required",
    "error":         str | None,
  }
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from . import sql_validator
from .snowflake_db import run_query_df

logger = logging.getLogger(__name__)

_CHAT         = Path(__file__).resolve().parent
_CATALOG_PATH = _CHAT / "schema_catalog.json"
_PROMPT_PATH  = _CHAT / "sql_prompt.md"
_LOG_PATH     = _CHAT / "sql_requests.jsonl"

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
_SQL_MODEL = "llama-3.3-70b-versatile"

# Hard SQL-signal patterns — if any match, not_required fast-exit is suppressed
_HARD_SQL_SIGNALS = re.compile(
    r"\b(trend|historical|history|over\s+time|over\s+the\s+past"
    r"|last\s+year|last\s+quarter"
    r"|compare|versus|ranking|rank"
    r"|january|february|march|april|may|june"
    r"|july|august|september|october|november|december"
    r"|quarter|q1|q2|q3|q4|in\s+\d{4}|since\s+\d{4}"
    r"|how\s+many\s+total|total\s+across|across\s+all|sum\s+of"
    r"|breakdown|by\s+ward|per\s+ward|which\s+ward"
    r"|highest|lowest|most|least"
    r")\b",
    re.IGNORECASE,
)


# ─── Schema summary ───────────────────────────────────────────────────────────

def _derive_schema_summary(catalog: dict) -> str:
    lines = []
    for t in catalog.get("tables", []):
        lines.append(f"TABLE: {t['alias']} ({t['table']})")
        lines.append(f"  Scope: {t.get('facility_scope', 'unknown')} | Grain: {t.get('grain', '')}")
        cols = t.get("columns", {})
        col_str = ", ".join(
            f"{c} ({m.get('type', '?')})" + (f" — {m['note']}" if m.get("note") else "")
            for c, m in cols.items()
        )
        lines.append(f"  Columns: {col_str}")
        mandatory = t.get("mandatory_filters", [])
        if mandatory:
            clauses = "; ".join(
                f"{f['column']} {f['operator']} ({f.get('when', '')})" for f in mandatory
            )
            lines.append(f"  Mandatory filters: {clauses}")
        conditional = t.get("conditional_filters", [])
        for cf in conditional:
            lines.append(f"  Conditional [{cf.get('context', '')}]: {cf.get('filter', '')}")
        for note in t.get("advisory_notes", []):
            lines.append(f"  Advisory: {note}")
        lines.append("")
    return "\n".join(lines)


# ─── Prompt assembly ──────────────────────────────────────────────────────────

def _build_sql_prompt(question: str, schema_summary: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    # Strip comment header lines (lines starting with #)
    body_lines = [l for l in template.splitlines() if not l.startswith("#")]
    body = "\n".join(body_lines).strip()
    return body.replace("{schema_summary}", schema_summary).replace("{question}", question)


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str) -> str:
    groq_key = os.getenv("GROQ_API", "")
    if not groq_key:
        raise RuntimeError("GROQ_API key not configured")
    with httpx.Client() as client:
        resp = client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model":       _SQL_MODEL,
                "messages":    [{"role": "user", "content": system_prompt}],
                "max_tokens":  512,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ─── Audit logging ────────────────────────────────────────────────────────────

def _log(record: dict) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("sql_executor: audit log write failed — %s", exc)


# ─── SQLResult builder ────────────────────────────────────────────────────────

def _result(
    question: str,
    sql: Optional[str],
    rows: int,
    columns: list,
    data: list,
    confidence: str,
    error: Optional[str] = None,
) -> dict:
    return {
        "question":      question,
        "generated_sql": sql,
        "returned_rows": rows,
        "columns":       columns,
        "data":          data,
        "confidence":    confidence,
        "error":         error,
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def execute(question: str, route_result: dict) -> dict:
    """
    Generate and execute SQL for the given question.

    Args:
        question:     Raw user question.
        route_result: Output of router.route() — must include use_snapshot, use_sql, matches.

    Returns:
        SQLResult dict. confidence == "not_required" means SQL was skipped.
    """
    t_start = time.monotonic()

    # ── Catalog + version ─────────────────────────────────────────────────────
    try:
        catalog_bytes   = _CATALOG_PATH.read_bytes()
        catalog         = json.loads(catalog_bytes.decode("utf-8"))
        catalog_hash    = hashlib.md5(catalog_bytes).hexdigest()
        catalog_version = catalog.get("version", "unknown")
    except Exception as exc:
        err = f"catalog_load_error:{exc}"
        _log({"ts": datetime.now(timezone.utc).isoformat(), "question": question,
              "outcome": "failed", "error": err, "elapsed_s": 0})
        return _result(question, None, 0, [], [], "failed", err)

    schema_summary = _derive_schema_summary(catalog)

    # ── not_required fast-exit ─────────────────────────────────────────────────
    # Skip SQL when snapshot matches are present and clean, and question has no
    # hard SQL signals (specific dates, breakdowns, historical totals).
    use_snapshot = route_result.get("use_snapshot", False)
    matches      = route_result.get("matches", [])
    if use_snapshot and matches and not _HARD_SQL_SIGNALS.search(question):
        all_clean = all(
            m.get("snapshot") is not None and m["snapshot"].get("fetch_ok")
            for m in matches
        )
        if all_clean:
            _log({
                "ts":              datetime.now(timezone.utc).isoformat(),
                "question":        question,
                "outcome":         "not_required",
                "catalog_version": catalog_version,
                "catalog_hash":    catalog_hash,
                "elapsed_s":       round(time.monotonic() - t_start, 3),
            })
            return _result(question, None, 0, [], [], "not_required")

    # ── Build generation prompt ────────────────────────────────────────────────
    try:
        prompt_text = _build_sql_prompt(question, schema_summary)
    except Exception as exc:
        err = f"prompt_build_error:{exc}"
        _log({"ts": datetime.now(timezone.utc).isoformat(), "question": question,
              "outcome": "failed", "error": err})
        return _result(question, None, 0, [], [], "failed", err)

    # ── LLM: generate SQL (with one retry on validation failure) ──────────────
    initial_sql     = None
    regenerated_sql = None
    validation_result = None
    rejection_reason  = None
    final_sql         = None

    for attempt in range(2):
        try:
            if attempt == 0:
                raw_sql = _call_llm(prompt_text)
            else:
                # Retry: include validation errors in the prompt
                retry_prompt = (
                    prompt_text
                    + f"\n\n## Previous attempt was rejected\nReason: {rejection_reason}\n"
                    + "Rewrite the SQL fixing the listed issue."
                )
                raw_sql = _call_llm(retry_prompt)

            # Strip markdown fences if the model wrapped the SQL
            raw_sql = re.sub(r"^```\w*\n?", "", raw_sql.strip(), flags=re.MULTILINE)
            raw_sql = re.sub(r"```$", "", raw_sql.strip())
            raw_sql = raw_sql.strip()

        except Exception as exc:
            err = f"llm_error:{exc}"
            _log({
                "ts":              datetime.now(timezone.utc).isoformat(),
                "question":        question,
                "outcome":         "failed",
                "error":           err,
                "catalog_version": catalog_version,
                "catalog_hash":    catalog_hash,
                "elapsed_s":       round(time.monotonic() - t_start, 3),
            })
            return _result(question, None, 0, [], [], "failed", err)

        # Model signalled it cannot answer
        if raw_sql.upper().startswith("NO_SQL:"):
            reason = raw_sql[7:].strip()
            _log({
                "ts":              datetime.now(timezone.utc).isoformat(),
                "question":        question,
                "outcome":         "no_sql",
                "no_sql_reason":   reason,
                "catalog_version": catalog_version,
                "catalog_hash":    catalog_hash,
                "elapsed_s":       round(time.monotonic() - t_start, 3),
            })
            return _result(question, None, 0, [], [], "failed", f"no_sql:{reason}")

        if attempt == 0:
            initial_sql = raw_sql
        else:
            regenerated_sql = raw_sql

        # Validate
        validation_result = sql_validator.validate(raw_sql)
        if validation_result["valid"]:
            final_sql = raw_sql
            break

        rejection_reason = "; ".join(validation_result["errors"])
        if attempt == 0:
            logger.info("sql_executor: validation failed (attempt 1) — %s. Retrying.", rejection_reason)
        else:
            # Second failure — give up
            _log({
                "ts":               datetime.now(timezone.utc).isoformat(),
                "question":         question,
                "outcome":          "validation_failed_after_retry",
                "initial_sql":      initial_sql,
                "regenerated_sql":  regenerated_sql,
                "validation_errors": validation_result["errors"],
                "catalog_version":  catalog_version,
                "catalog_hash":     catalog_hash,
                "elapsed_s":        round(time.monotonic() - t_start, 3),
            })
            return _result(
                question, regenerated_sql or initial_sql, 0, [], [],
                "failed", f"validation_failed:{rejection_reason}"
            )

    if final_sql is None:
        return _result(question, initial_sql, 0, [], [], "failed", "no_valid_sql_generated")

    # ── Execute on Snowflake ──────────────────────────────────────────────────
    try:
        df = run_query_df(final_sql)
        df.columns = df.columns.str.lower()

        rows    = len(df)
        columns = list(df.columns)
        data    = df.head(200).to_dict(orient="records")

        # Confidence: validated if rows > 0, partial if empty (valid SQL, no data)
        confidence = "validated" if rows > 0 else "partial"

        elapsed = round(time.monotonic() - t_start, 3)
        _log({
            "ts":               datetime.now(timezone.utc).isoformat(),
            "question":         question,
            "outcome":          confidence,
            "initial_sql":      initial_sql,
            "regenerated_sql":  regenerated_sql,
            "final_sql":        final_sql,
            "validation_errors": [],
            "validation_warnings": validation_result.get("warnings", []),
            "tables_used":      validation_result.get("tables_used", []),
            "returned_rows":    rows,
            "catalog_version":  catalog_version,
            "catalog_hash":     catalog_hash,
            "elapsed_s":        elapsed,
        })
        return _result(question, final_sql, rows, columns, data, confidence)

    except Exception as exc:
        err = f"execution_error:{exc}"
        elapsed = round(time.monotonic() - t_start, 3)
        _log({
            "ts":               datetime.now(timezone.utc).isoformat(),
            "question":         question,
            "outcome":          "execution_failed",
            "final_sql":        final_sql,
            "error":            str(exc),
            "catalog_version":  catalog_version,
            "catalog_hash":     catalog_hash,
            "elapsed_s":        elapsed,
        })
        logger.error("sql_executor: execution error — %s", exc, exc_info=True)
        return _result(question, final_sql, 0, [], [], "failed", err)
