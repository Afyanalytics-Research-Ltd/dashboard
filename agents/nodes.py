"""
LangGraph node functions — shared utilities + the human-in-the-loop
fallback/resume path.

The happy-path nodes (plan_intent, retrieve_candidates, generate_cube_query,
validate_query, execute_query, explain_result) live in nodes_intent.py /
nodes_query.py / nodes_explain.py. This module keeps:
  - _openai() / _recent_history() / _context_hint() — shared by every node
    module above (imported from here, not duplicated).
  - fallback_notifier / re_classify / auto_notify_user — the deep fallback
    for questions the retrieval pipeline genuinely can't resolve. Unchanged
    behavior: email the analytics team, interrupt() until a human resumes
    via POST /api/resume/, then re-run the query with the now-known metric.

Graph topology (see agents/graph.py):
    plan_intent → retrieve_candidates ─┬─[confident match]──→ generate_cube_query → validate_query → execute_query → explain_result
                                        └─[nothing relevant]─→ fallback_notifier ─[interrupt]→ re_classify ──────────────────────┘
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy

import httpx
from django.conf import settings
from django.core.mail import send_mail
from langgraph.types import interrupt
from openai import OpenAI

from .catalog import get_by_id, reload as reload_catalog
from .state import AgentState
from .schema_validation import (
    valid_members_for as _valid_members_for,
    valid_time_members_for as _valid_time_members_for,
    validate_filters as _validate_filters,
)

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


CONFIDENCE_THRESHOLD = 0.75  # below this → fallback

HISTORY_TURNS = 8  # user+assistant exchanges fed as LLM context per call


def _recent_history(state: AgentState, *, include_current: bool = True) -> list[dict]:
    """
    Last few turns of state["messages"], formatted for a chat-completions
    `messages` list. plan_intent runs before the current question has been
    appended, so include_current is moot there; explain_result runs after
    it (appended by plan_intent), so it passes include_current=False to
    avoid echoing the current question twice — once from history, once
    from the explicit prompt built in that node.
    """
    history = state.get("messages") or []
    if not include_current and history and history[-1].get("role") == "user":
        history = history[:-1]
    return history[-HISTORY_TURNS * 2:]


def _context_hint(state: AgentState) -> str:
    """
    Deterministic anchor for elliptical follow-ups: the last metric that
    actually matched in this thread. Unlike raw message history (which the
    model has to parse for itself), this is handed over pre-resolved so a
    question like "and for last month?" doesn't depend on the model
    correctly inferring the subject from prior free text alone.
    """
    last_metric = state.get("last_matched_metric")
    if not last_metric:
        return ""
    return (
        f"\nMost recent metric matched in this conversation: '{last_metric['id']}' "
        f"({last_metric['name']}). A follow-up question with no new subject named "
        f"almost certainly still means this metric.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node — fallback_notifier  ← contains the interrupt()
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
# Node — re_classify  (resume path only)
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
    allowed_members = _valid_members_for(base)
    time_members = _valid_time_members_for(base)
    filter_prompt = (
        f"Extract any dimension filters from this question for the metric "
        f"'{base['name']}'. Only use these exact fields as a filter's \"member\" — "
        f"never invent or paraphrase one: {sorted(allowed_members)}. "
        f"Date-range operators (inDateRange, notInDateRange, beforeDate, afterDate) "
        f"may ONLY target one of these date fields: {sorted(time_members) or 'none — do not emit a date-range filter for this metric'}. "
        f"Respond with JSON: {{\"filters\": [...]}}. "
        f"Return an empty list if no filters are stated, or if none of the fields above fit. No markdown."
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
    matched["cube_query"]["filters"].extend(
        _validate_filters(extra_filters, allowed_members=allowed_members, time_members=time_members)
    )

    logger.info("re_classify: using metric=%s with %d extra filters", metric_id, len(extra_filters))

    return {
        "matched_metric": matched,
        "cube_query": matched["cube_query"],
        "classification_confidence": 1.0,  # team confirmed this metric
        "last_matched_metric": matched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node — auto_notify_user  (resume path only)
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
