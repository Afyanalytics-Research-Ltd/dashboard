"""
LangGraph node functions.

Each function receives AgentState and returns a dict of state updates.

Node order (happy path):
    classify_intent → execute_query → format_result → END

Fallback path (no match):
    classify_intent → fallback_notifier ─[interrupt]─ → re_classify → execute_query → format_result → auto_notify_user → END
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any

import httpx
from django.conf import settings
from django.core.mail import send_mail
from langgraph.types import interrupt
from openai import OpenAI

from . import charts
from .catalog import as_context, get_all, get_by_id, reload as reload_catalog
from .cube_client import run_query
from .state import AgentState
from .facility import resolve_facility, inject_facility_filter

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


CONFIDENCE_THRESHOLD = 0.75  # below this → fallback

# Valid Cube.dev filter operators
VALID_CUBE_OPERATORS = {
    "equals", "notEquals", "contains", "notContains", "startsWith", "endsWith",
    "gt", "gte", "lt", "lte", "set", "notSet",
    "inDateRange", "notInDateRange", "beforeDate", "afterDate",
}


def _validate_filters(filters: list[dict]) -> list[dict]:
    """
    Strip out any filters that are malformed or use unsupported operators.
    Prevents Cube 400 errors caused by LLM hallucinating operator names.
    """
    valid = []
    for f in filters:
        member = f.get("member", "")
        operator = f.get("operator", "")
        values = f.get("values")

        if not member or not operator:
            logger.warning("_validate_filters: dropping filter missing member/operator: %s", f)
            continue
        # Member must be in CubeName.fieldName format — bare words like "date" are invalid
        if "." not in member:
            logger.warning(
                "_validate_filters: dropping filter — member '%s' is not in Cube.member format", member
            )
            continue
        if operator not in VALID_CUBE_OPERATORS:
            logger.warning("_validate_filters: dropping filter with invalid operator '%s': %s", operator, f)
            continue
        # notSet / set don't require values
        if operator not in ("set", "notSet") and not values:
            logger.warning("_validate_filters: dropping filter missing values: %s", f)
            continue

        valid.append({"member": member, "operator": operator, "values": values or []})
    return valid


def _promote_date_filters(query: dict, filters: list[dict]) -> tuple[dict, list[dict]]:
    """
    Date range filters whose member matches an existing timeDimension should be
    expressed as timeDimension.dateRange, NOT as a separate filter — Cube rejects
    the latter when a timeDimension for that member already exists.

    Returns (updated_query, remaining_filters).
    """
    date_ops = {"inDateRange", "notInDateRange", "beforeDate", "afterDate"}
    time_dims = [dict(td) for td in query.get("timeDimensions", [])]
    remaining = []

    for f in filters:
        if f["operator"] not in date_ops:
            remaining.append(f)
            continue

        member = f["member"]
        matched = False
        for td in time_dims:
            if td.get("dimension") == member:
                if f["operator"] == "inDateRange":
                    td["dateRange"] = f["values"]  # e.g. ["2023-09-01", "2023-09-30"]
                matched = True
                logger.info("_promote_date_filters: moved %s filter to timeDimension.dateRange", member)
                break

        if not matched:
            # No matching timeDimension — keep as a regular filter
            remaining.append(f)

    updated_query = {**query, "timeDimensions": time_dims}
    return updated_query, remaining


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — classify_intent
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """\
You are a semantic router for an analytics platform.
 
Your job is to match a user's natural-language question to ONE of the predefined
metrics below, and extract any dimension filters implied by the question
(e.g. date ranges, facility names, product names).
 
Predefined metrics:
{catalog}
 
CRITICAL: The valid metric_id values are ONLY the ids listed above.
You MUST pick from that exact list — do NOT invent or paraphrase an id.
If no metric fits, return null.
 
Respond with valid JSON only — no markdown, no explanation:
{{
  "metric_id": "<exact id from the list above, or null>",
  "confidence": <float 0.0–1.0>,
  "filters": [
    {{"member": "<CubeName.dimensionName>", "operator": "<operator>", "values": ["<val>"]}}
  ],
  "reasoning": "<one sentence explaining why this metric was chosen or why no match>"
}}
 
Allowed operators (use exactly as written):
  equals, notEquals, contains, notContains, startsWith, endsWith,
  gt, gte, lt, lte, set, notSet,
  inDateRange, notInDateRange, beforeDate, afterDate
 
Rules:
- metric_id MUST be one of the ids shown in the catalog above, or null.
- If the question clearly maps to a metric, set confidence ≥ 0.8.
- If it partially matches but you are unsure, set confidence 0.5–0.79.
- If there is no reasonable match, set metric_id to null and confidence < 0.5.
- Only include filters explicitly stated or strongly implied by the question.
- Return an empty filters array [] if no filters apply.
- For date filters use inDateRange with values ["YYYY-MM-DD", "YYYY-MM-DD"].
- For facility filters use member "Dispensing.facility" with values like ["KISUMU"].
"""

def classify_intent(state: AgentState) -> dict:
    catalog_text = as_context()
    logging.warning(f'catalog_text: {catalog_text}')

    prompt = CLASSIFY_SYSTEM.format(catalog=catalog_text)

    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed: dict = json.loads(raw)
    # import pdb;pdb.set_trace()
    metric_id = parsed.get("metric_id")
    confidence = float(parsed.get("confidence", 0.0))
    filters = parsed.get("filters", [])

    matched_metric = None
    # import pdb;pdb.set_trace()
    if metric_id and confidence >= CONFIDENCE_THRESHOLD:
        base = get_by_id(metric_id)
        if base:
            matched_metric = deepcopy(base)
            # Merge validated LLM-extracted filters into the predefined query
            matched_metric["cube_query"]["filters"].extend(_validate_filters(filters))

    logger.info(
        "classify_intent: metric_id=%s confidence=%.2f matched=%s",
        metric_id,
        confidence,
        bool(matched_metric),
    )

    return {
        "matched_metric": matched_metric,
        "classification_confidence": confidence,
        "fallback_reason": None if matched_metric else parsed.get("reasoning"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — execute_query  (shared by happy path and resume path)
# ─────────────────────────────────────────────────────────────────────────────

def execute_query(state: AgentState) -> dict:
    metric = state["matched_metric"]
    query = dict(metric["cube_query"])  # shallow copy — don't mutate state

    # Promote inDateRange filters → timeDimension.dateRange before sending
    user_facility = state.get("user_facility") or resolve_facility(state["user_id"])
    if user_facility:
        query = inject_facility_filter(query, user_facility)  # ← filter added here
    
    filters = query.get("filters", [])
    if filters:
        query, filters = _promote_date_filters(query, filters)
        query["filters"] = filters

    logger.info("execute_query: metric=%s query=%s", metric["id"], query)

    try:
        result = run_query(query)
    except httpx.HTTPStatusError as exc:
        logger.error("Cube API error: %s — %s", exc.response.status_code, exc.response.text)
        raise
    except httpx.TimeoutException:
        logger.error("Cube API timed out for metric %s", metric["id"])
        raise

    return {
        "cube_query": query,
        "raw_result": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — format_result  (shared)
# ─────────────────────────────────────────────────────────────────────────────

FORMAT_SYSTEM = """\
You are a data analyst assistant. Given a user question and a raw Cube.dev API
response, produce a concise, helpful answer.

Respond with valid JSON only:
{{
  "summary": "<1–3 sentence plain-English answer>",
  "data": <the relevant rows/values from the raw result, cleaned up>,
  "metric_name": "<human-readable metric name>"
}}
"""


def format_result(state: AgentState) -> dict:
    raw = state["raw_result"]
    question = state["question"]
    metric = state["matched_metric"]

    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FORMAT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Metric: {metric['name']}\n\n"
                    f"Raw result:\n{json.dumps(raw, indent=2)}"
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    formatted = json.loads(response.choices[0].message.content)
    formatted["metric_id"] = metric["id"]
    formatted["thread_id"] = state["thread_id"]
    formatted["row_count"] = len(raw.get("data") or [])
    # Offer a chart rather than building one eagerly — most results are small
    # enough that the text summary already suffices, and building unwanted
    # charts wastes compute. The actual image is only rendered later, on
    # confirmation, via charts.get_chart_for_thread(thread_id).
    formatted["chart_offer"] = charts.is_heavy(raw)

    return {"formatted_result": formatted}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — fallback_notifier  ← contains the interrupt()
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_USER_MESSAGE = (
    "Hello! 👋 Thank you for your question.\n\n"
    "The metric you're asking about isn't available in our system yet, but "
    "we've already notified our analytics team and they're working on it. "
    "You'll receive the answer here as soon as it's ready.\n\n"
    "We appreciate your patience! 🙏"
)


def _send_whatsapp(phone: str, message: str) -> None:
    """
    Send a WhatsApp message via Whapi (https://whapi.cloud/).

    Required settings:
        WHAPI_TOKEN   — your Whapi channel token
        WHAPI_URL     — channel gateway URL, e.g. https://gate.whapi.cloud
    """
    token = getattr(settings, "WHAPI_TOKEN", os.getenv("WHAPI_TOKEN", ""))
    base_url = getattr(settings, "WHAPI_URL", os.getenv("WHAPI_URL", "https://gate.whapi.cloud")).rstrip("/")

    if not token:
        logger.warning("_send_whatsapp: WHAPI_TOKEN not configured — skipping WhatsApp send.")
        return

    # Whapi expects phone as "{number}@s.whatsapp.net" (no + prefix)
    phone_clean = phone.lstrip("+").replace(" ", "")
    to = f"{phone_clean}@s.whatsapp.net"

    payload = {"to": to, "body": message}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{base_url}/messages/text", json=payload, headers=headers)
            resp.raise_for_status()
        logger.info("_send_whatsapp: message sent to %s", phone)
    except Exception as exc:
        logger.error("_send_whatsapp: failed to send to %s — %s", phone, exc)


def _send_whatsapp_image(phone: str, image_base64: str, caption: str = "", mime: str = "image/png") -> None:
    """
    Send an image message via Whapi (https://whapi.cloud/).

    NOT YET VERIFIED against a live Whapi channel — unlike _send_whatsapp's
    /messages/text call (already exercised in production here), this repo
    has no prior working example of Whapi's media-message payload shape.
    Send one real chart to a test number before relying on this — if it
    fails, check Whapi's current /messages/image (or /messages/media) schema
    and adjust the payload below to match.

    Required settings: WHAPI_TOKEN, WHAPI_URL (same as _send_whatsapp).
    """
    token = getattr(settings, "WHAPI_TOKEN", os.getenv("WHAPI_TOKEN", ""))
    base_url = getattr(settings, "WHAPI_URL", os.getenv("WHAPI_URL", "https://gate.whapi.cloud")).rstrip("/")

    if not token:
        logger.warning("_send_whatsapp_image: WHAPI_TOKEN not configured — skipping image send.")
        return

    phone_clean = phone.lstrip("+").replace(" ", "")
    to = f"{phone_clean}@s.whatsapp.net"

    payload = {
        "to": to,
        "media": f"data:{mime};base64,{image_base64}",
        "caption": caption,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.post(f"{base_url}/messages/image", json=payload, headers=headers)
            resp.raise_for_status()
        logger.info("_send_whatsapp_image: chart sent to %s", phone)
    except Exception as exc:
        logger.error("_send_whatsapp_image: failed to send to %s — %s", phone, exc)


def _ping_callback(callback_url: str, state: AgentState, user_message: str) -> None:
    """
    Immediately POST a 'pending' status to the client's callback_url
    so the caller knows the request was received and is being worked on.
    """
    if not callback_url:
        return

    payload = {
        "thread_id": state["thread_id"],
        "user_id": state["user_id"],
        "question": state["question"],
        "status": "pending",
        "message": user_message,
    }

    try:
        with httpx.Client(timeout=8) as client:
            resp = client.post(callback_url, json=payload)
            resp.raise_for_status()
        logger.info("_ping_callback: pending status sent to %s", callback_url)
    except Exception as exc:
        logger.error("_ping_callback: failed to reach %s — %s", callback_url, exc)


def fallback_notifier(state: AgentState) -> dict:
    """
    Phase 1:
      1. Email the analytics team with the unmet request.
      2. Send a polite WhatsApp message to the user via Whapi.
      3. POST a 'pending' status to the client's callback_url.
      4. Call interrupt() — graph suspends until analytics team resumes.

    Phase 2 (after resume):
      - interrupt() returns the resume payload from Command(resume=…).
      - We store it in state and fall through to re_classify.
    """
    analytics_email = getattr(settings, "ANALYTICS_TEAM_EMAIL", "analytics@example.com")
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", "noreply@example.com")

    # ── 1. Email analytics team ───────────────────────────────────────────
    subject = f"[Analytics Request] New metric needed: \"{state['question'][:60]}\""
    body = (
        f"A user has asked a question that does not match any predefined metric.\n\n"
        f"Question  : {state['question']}\n"
        f"User ID   : {state['user_id']}\n"
        f"User Phone: {state.get('user_phone') or 'not provided'}\n"
        f"Thread ID : {state['thread_id']}\n"
        f"Confidence: {state['classification_confidence']:.0%}\n"
        f"Reason    : {state.get('fallback_reason', 'No close match found')}\n\n"
        f"Once you have added the metric to catalog/metrics.yaml, trigger the resume:\n\n"
        f"  POST /api/resume/\n"
        f"  {{\n"
        f"    \"thread_id\": \"{state['thread_id']}\",\n"
        f"    \"metric_id\": \"<your_new_metric_id>\"\n"
        f"  }}\n\n"
        f"The user will be notified automatically via WhatsApp and webhook once the query runs."
    )

    try:
        send_mail(subject, body, from_email, [analytics_email], fail_silently=False)
        logger.info("fallback_notifier: email sent to %s", analytics_email)
    except Exception as exc:
        logger.error("fallback_notifier: failed to send email — %s", exc)

    # ── 2. WhatsApp the user ──────────────────────────────────────────────
    user_phone = state.get("user_phone")
    if user_phone:
        _send_whatsapp(user_phone, FALLBACK_USER_MESSAGE)
    else:
        logger.info("fallback_notifier: no user_phone in state — skipping WhatsApp.")

    # ── 3. Ping callback_url with pending status ───────────────────────────
    _ping_callback(state.get("callback_url", ""), state, FALLBACK_USER_MESSAGE)

    # ── 4. Suspend — graph sleeps here until Command(resume=…) is called ──
    logger.info("fallback_notifier: suspending thread %s", state["thread_id"])
    resume_payload = interrupt(
        {
            "status": "waiting_for_metric",
            "thread_id": state["thread_id"],
            "question": state["question"],
            "user_message": FALLBACK_USER_MESSAGE,
        }
    )

    # ── Phase 2: arrived here after human triggered resume ────────────────
    logger.info("fallback_notifier: resumed with payload=%s", resume_payload)
    return {
        "is_resumed": True,
        "resume_data": resume_payload,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — re_classify  (resume path only)
# ─────────────────────────────────────────────────────────────────────────────

def re_classify(state: AgentState) -> dict:
    """
    After resume, the analytics team has supplied a metric_id.
    Reload the catalog (busts the lru_cache so the new metric is visible),
    then use the known metric_id + original question to build the query.

    We still run a lightweight LLM call to extract filters from the original
    question (date ranges, dimension values, etc.).
    """
    reload_catalog()  # pick up newly added metric

    resume_data = state.get("resume_data") or {}
    metric_id = resume_data.get("metric_id")

    if not metric_id:
        raise ValueError("resume_data must contain 'metric_id'")

    base = get_by_id(metric_id)
    if not base:
        raise ValueError(
            f"Metric '{metric_id}' not found in catalog after reload. "
            "Make sure it was added to catalog/metrics.yaml before resuming."
        )

    # Extract filters from original question for the specific metric
    filter_prompt = (
        f"Extract any dimension filters from this question for the metric "
        f"'{base['name']}' (Cube cube: {list(set(m.split('.')[0] for m in base['cube_query'].get('measures', [])))}). "
        f"Respond with JSON: {{\"filters\": [...]}}. "
        f"Return an empty list if no filters are stated. No markdown."
    )

    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": filter_prompt},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    extra_filters = json.loads(response.choices[0].message.content).get("filters", [])

    matched = deepcopy(base)
    matched["cube_query"]["filters"].extend(_validate_filters(extra_filters))

    logger.info("re_classify: using metric=%s with %d extra filters", metric_id, len(extra_filters))

    return {
        "matched_metric": matched,
        "classification_confidence": 1.0,  # team confirmed this metric
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — auto_notify_user  (resume path only)
# ─────────────────────────────────────────────────────────────────────────────

def auto_notify_user(state: AgentState) -> dict:
    """
    Phase 2 completion:
      1. POST the formatted result to the client's callback_url.
      2. Send a WhatsApp message to the user letting them know their metric is ready.
    """
    formatted = state.get("formatted_result") or {}
    metric_name = formatted.get("metric_name", "your requested metric")
    summary = formatted.get("summary", "")

    # ── 1. Webhook ────────────────────────────────────────────────────────
    callback_url = state.get("callback_url")
    if callback_url:
        payload = {
            "thread_id": state["thread_id"],
            "user_id": state["user_id"],
            "question": state["question"],
            "result": formatted,
            "status": "completed",
        }
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(callback_url, json=payload)
                resp.raise_for_status()
            logger.info("auto_notify_user: webhook delivered to %s", callback_url)
        except Exception as exc:
            logger.error("auto_notify_user: webhook delivery failed — %s", exc)
    else:
        logger.warning("auto_notify_user: no callback_url — skipping webhook.")

    # ── 2. WhatsApp the user ──────────────────────────────────────────────
    user_phone = state.get("user_phone")
    if user_phone:
        ready_message = (
            f"✅ Great news! Your metric *{metric_name}* is now ready.\n\n"
            f"{summary}\n\n"
            f"You can now ask the same question again and you'll get the full result instantly."
        )
        _send_whatsapp(user_phone, ready_message)

    return {}