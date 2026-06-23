"""
Django API views.

POST /api/query/      — Start a query run (Phase 1).
POST /api/whatsapp/   — Whapi webhook: WhatsApp messages enter here.
POST /api/resume/     — Analytics team triggers Phase 2 after adding a metric.
"""

from __future__ import annotations

import json
import logging
import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langgraph.types import Command

from .graph import graph
from .facility import resolve_facility, resolve_facility_from_user

logger = logging.getLogger(__name__)


def _build_initial_state(
    question: str,
    user_id: str,
    callback_url: str,
    user_phone: str | None = None,
) -> dict:
    thread_id = str(uuid.uuid4())
    # Resolve facility upfront — try user_id first, then phone number.
    # This means both REST and WhatsApp callers get row-level filtering.
    user_facility = resolve_facility(user_id, phone=user_phone)
    if user_facility:
        logger.info("_build_initial_state: user=%s → facility=%s", user_id, user_facility)
    else:
        logger.info("_build_initial_state: user=%s → no facility restriction", user_id)
    return thread_id, {
        "question": question,
        "user_id": user_id,
        "user_phone": user_phone,
        "callback_url": callback_url,
        "thread_id": thread_id,
        "user_facility": user_facility,
        "matched_metric": None,
        "classification_confidence": 0.0,
        "cube_query": None,
        "raw_result": None,
        "formatted_result": None,
        "is_resumed": False,
        "fallback_reason": None,
        "resume_data": None,
    }


def _run_graph(initial_state: dict, thread_id: str):
    """Invoke the graph and return (output, error_response)."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        output = graph.invoke(initial_state, config=config)
        return output, None
    except Exception as exc:
        logger.exception("graph.invoke failed for thread %s", thread_id)
        return None, JsonResponse({"error": str(exc)}, status=500)


def _interrupt_message(output: dict) -> str | None:
    """Return the interrupt user_message if the graph suspended, else None."""
    interrupts = output.get("__interrupt__")
    if not interrupts:
        return None
    interrupt_value = (
        interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
    )
    return interrupt_value.get(
        "user_message",
        "Your request is being processed. You'll be notified when it's ready.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/query/
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def query(request):
    """
    Start a query run via JSON API.

    Request body:
        {
            "question":     "What was our revenue last month?",
            "user_id":      "user_abc123",           (optional if authenticated)
            "callback_url": "https://your-app.com/webhooks/analytics-result",
            "user_phone":   "254712345678"           (optional, for WhatsApp reply)
        }

    Response 200 — metric found:
        { "status": "completed", "thread_id": "…", "result": { … } }

    Response 202 — metric not found, graph suspended:
        { "status": "pending", "thread_id": "…", "message": "…" }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    question     = (body.get("question")     or "").strip()
    callback_url = (body.get("callback_url") or "").strip()
    user_phone   = (body.get("user_phone")   or "").strip() or None

    # user_id: prefer body value, fall back to authenticated user's username
    user_id = (body.get("user_id") or "").strip()
    if not user_id and request.user.is_authenticated:
        user_id = request.user.username

    if not question:
        return JsonResponse({"error": "'question' is required."}, status=400)
    if not user_id:
        return JsonResponse({"error": "'user_id' is required."}, status=400)
    if not callback_url:
        logger.warning("query: no callback_url provided for user=%s", user_id)

    # Facility resolution: use authenticated User object if available (fastest),
    # otherwise fall through string-based resolve_facility.
    if request.user.is_authenticated:
        user_facility = resolve_facility_from_user(request.user)
    else:
        user_facility = resolve_facility(user_id, phone=user_phone)

    thread_id = str(uuid.uuid4())
    initial_state = {
        "question":               question,
        "user_id":                user_id,
        "user_phone":             user_phone,
        "callback_url":           callback_url or "none://unset",
        "thread_id":              thread_id,
        "user_facility":          user_facility,
        "matched_metric":         None,
        "classification_confidence": 0.0,
        "cube_query":             None,
        "raw_result":             None,
        "formatted_result":       None,
        "is_resumed":             False,
        "fallback_reason":        None,
        "resume_data":            None,
    }

    logger.info(
        "query: user=%s facility=%s question=%r",
        user_id, user_facility, question,
    )
    output, err = _run_graph(initial_state, thread_id)
    if err:
        return err

    user_message = _interrupt_message(output)
    if user_message:
        return JsonResponse(
            {"status": "pending", "thread_id": thread_id, "message": user_message},
            status=202,
        )

    return JsonResponse(
        {"status": "completed", "thread_id": thread_id, "result": output.get("formatted_result")},
        status=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/whatsapp/   (Whapi webhook)
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def whatsapp_webhook(request):
    """
    Receives Whapi webhook events and routes them through the agent.

    Whapi POSTs a payload like:
        {
            "messages": [{
                "from":      "254700701209",
                "chat_id":   "254700701209@s.whatsapp.net",
                "from_name": "lutherlunyamwi",
                "from_me":   false,
                "type":      "text",
                "text":      {"body": "What was the revenue?"},
                ...
            }],
            "event":      {"type": "messages", "event": "post"},
            "channel_id": "NIGHTW-GAZZZ"
        }

    Results are delivered back to the user via WhatsApp (auto_notify_user node),
    so no callback_url is required — we use an internal placeholder.
    Whapi expects a fast 200 ack regardless of processing outcome.
    """
    # Whapi may send a GET verification ping on webhook registration
    if request.method == "GET":
        return JsonResponse({"status": "ok"})

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    messages = body.get("messages") or []
    if not messages:
        # Status updates, receipts, etc. — nothing to process
        return JsonResponse({"status": "ignored"}, status=200)

    msg = messages[0]

    # Ignore our own outbound messages
    if msg.get("from_me"):
        return JsonResponse({"status": "ignored"}, status=200)

    # Only handle plain text messages
    if msg.get("type") != "text":
        return JsonResponse({"status": "ignored", "reason": "non-text message"}, status=200)

    text_body = (msg.get("text") or {}).get("body", "").strip()
    if not text_body:
        return JsonResponse({"status": "ignored", "reason": "empty text"}, status=200)

    phone   = msg.get("from", "")           # "254700701209"
    user_id = msg.get("chat_id") or phone   # "254700701209@s.whatsapp.net"

    thread_id, initial_state = _build_initial_state(
        question=text_body,
        user_id=user_id,
        callback_url="whatsapp://internal",  # no HTTP callback for WhatsApp queries
        user_phone=phone,
    )

    output, err = _run_graph(initial_state, thread_id)
    if err:
        # Still return 200 to Whapi — errors are logged server-side
        logger.error("whatsapp_webhook: graph error for thread %s", thread_id)
        return JsonResponse({"status": "error", "thread_id": thread_id}, status=200)

    interrupted = bool(output.get("__interrupt__"))
    return JsonResponse(
        {"status": "pending" if interrupted else "completed", "thread_id": thread_id},
        status=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/resume/
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def resume(request):
    """
    Analytics team calls this after adding a new metric to the catalog.

    Request body:
        {
            "thread_id": "<uuid from original query response>",
            "metric_id": "new_metric_id",
            "analyst":   "jane@afya.ai"   (optional, for audit logging)
        }

    Response 200: { "status": "resumed", "thread_id": "…" }
    Response 404: { "error": "Thread not found or already completed." }
    Response 400: { "error": "…" }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    thread_id = (body.get("thread_id") or "").strip()
    metric_id = (body.get("metric_id") or "").strip()
    analyst   = body.get("analyst", "unknown")

    if not thread_id:
        return JsonResponse({"error": "'thread_id' is required."}, status=400)
    if not metric_id:
        return JsonResponse({"error": "'metric_id' is required."}, status=400)

    config = {"configurable": {"thread_id": thread_id}}

    # Verify the thread exists and is currently suspended
    state_snapshot = graph.get_state(config)
    if state_snapshot is None:
        return JsonResponse(
            {"error": "Thread not found or already completed."}, status=404
        )

    logger.info(
        "resume: thread=%s metric_id=%s analyst=%s", thread_id, metric_id, analyst
    )

    try:
        graph.invoke(
            Command(resume={"metric_id": metric_id, "analyst": analyst}),
            config=config,
        )
    except Exception as exc:
        logger.exception("graph.invoke(resume) failed for thread %s", thread_id)
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({"status": "resumed", "thread_id": thread_id}, status=200)