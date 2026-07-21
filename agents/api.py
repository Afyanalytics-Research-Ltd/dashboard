"""
Django API views.

POST /api/query/      — Start a query run (Phase 1).
POST /api/whatsapp/   — Whapi webhook: WhatsApp messages enter here.
POST /api/resume/     — Analytics team triggers Phase 2 after adding a metric.
POST /api/visualize/  — Render a chart from a previously-run query's result.
"""

from __future__ import annotations

import json
import logging
import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langgraph.types import Command

from .charts import get_chart_for_thread
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

    # if request.method == "POST":
    #     return JsonResponse({"status":"ok"})
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

    phone   = msg.get("from", "")
    profile = None
    try:
        from authentication.models import UserProfile  # adjust import path if needed

        profile = (
            UserProfile.objects
            .select_related("facility")
            .filter(phone_number=phone)
            .first()
            or
            # Some systems store with country prefix; try both
            UserProfile.objects
            .select_related("facility")
            .filter(phone_number__endswith=phone[-9:])
            .first()
        )
    except Exception as exc:
        logger.debug("_lookup_by_phone(%s): %s", phone, exc)

    if not profile:
        logger.warning("whatsapp_webhook: no UserProfile found for phone=%s", phone)
        return JsonResponse({"status": "ignored", "reason": "unknown phone"}, status=200)

    user_id = str(profile.user.id)

    from .charts import get_chart_for_thread, is_affirmative_reply, wants_visualization
    from .models import WhatsAppChatState
    from .nodes import _send_whatsapp, _send_whatsapp_image

    # WhatsApp webhooks are stateless per-request (unlike the chat websocket,
    # which can hold "last result" on its own connection instance) — this
    # row is what lets a later "yes" / "can I get a graph" find its way back
    # to the right thread, mirroring self_service/consumers.py's in-memory
    # pending_chart_thread_id / last_metric_thread_id.
    chat_state, _created = WhatsAppChatState.objects.get_or_create(phone=phone)

    wants_chart = (
        (chat_state.chart_offer_pending and is_affirmative_reply(text_body))
        or (chat_state.last_metric_thread_id and wants_visualization(text_body))
    )

    if wants_chart and chat_state.last_metric_thread_id:
        chart, error = get_chart_for_thread(chat_state.last_metric_thread_id)
        chat_state.chart_offer_pending = False
        chat_state.save(update_fields=["chart_offer_pending", "updated_at"])

        if error:
            _send_whatsapp(phone=phone, message=error)
        else:
            _send_whatsapp_image(phone=phone, image_base64=chart["image_base64"], caption=chart["caption"])
        return JsonResponse({"status": "completed", "thread_id": chat_state.last_metric_thread_id}, status=200)

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

    if interrupted:
        user_message = _interrupt_message(output) or "Your request is being processed."
        _send_whatsapp(phone=phone, message=user_message)
        return JsonResponse({"status": "pending", "thread_id": thread_id}, status=200)

    formatted = output.get("formatted_result") or {}
    summary = formatted.get("summary", "")
    result_thread_id = formatted.get("thread_id") or thread_id
    chart = None

    # The user may have already asked for a chart in THIS SAME message
    # ("can I get a chart for patient admissions?") — that shouldn't need a
    # follow-up "yes" round-trip; render and send it right away.
    if wants_visualization(text_body):
        chart, chart_error = get_chart_for_thread(result_thread_id)
        if not chart and chart_error:
            summary += f"\n\n{chart_error}"
    elif formatted.get("chart_offer"):
        summary += (
            f"\n\n📊 This result has {formatted.get('row_count', 0)} rows — "
            f"reply *yes* if you'd like me to visualize it as a chart."
        )

    chat_state.last_metric_thread_id = result_thread_id or chat_state.last_metric_thread_id
    # Don't leave an offer pending if we already sent the chart in this reply.
    chat_state.chart_offer_pending = bool(formatted.get("chart_offer")) and not chart
    chat_state.save(update_fields=["last_metric_thread_id", "chart_offer_pending", "updated_at"])

    _send_whatsapp(phone=phone, message=summary)
    if chart:
        _send_whatsapp_image(phone=phone, image_base64=chart["image_base64"], caption=chart["caption"])
    return JsonResponse({"status": "completed", "thread_id": thread_id}, status=200)


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


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/visualize/
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def visualize(request):
    """
    Render a chart from a previously-run query's result.

    The result to chart isn't sent in the request body — it's pulled back out
    of the LangGraph checkpoint by thread_id, so callers don't need to
    resubmit the (possibly large) raw Cube result themselves. Every
    /api/query/ response (and every chat-agent reply) already carries
    "thread_id", "row_count" and "chart_offer" in its result — check
    chart_offer before calling this, so users aren't shown a chart they
    never asked to see.

    Request body:
        {"thread_id": "<uuid from a prior /api/query/ or chat response>"}

    Response 200:
        {"status": "ok", "chart": {"image_base64": "...", "mime": "image/png", "caption": "..."}}
    Response 404: thread not found / expired (MemorySaver checkpoints don't survive a restart).
    Response 422: the result's shape doesn't suit a simple chart (e.g. a single scalar KPI).
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    thread_id = (body.get("thread_id") or "").strip()
    if not thread_id:
        return JsonResponse({"error": "'thread_id' is required."}, status=400)

    chart, error = get_chart_for_thread(thread_id)
    if error:
        status = 404 if error == "Thread not found." else 422
        return JsonResponse({"error": error}, status=status)

    return JsonResponse({"status": "ok", "chart": chart}, status=200)