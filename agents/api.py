"""
Django API views.

POST /api/query/   — Start a query run (Phase 1).
POST /api/resume/  — Analytics team triggers Phase 2 after adding a metric.
"""

from __future__ import annotations

import logging
import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langgraph.types import Command

from .graph import graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/query/
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def query(request):
    """
    Start a query run.

    Request body (JSON):
        {
            "question":     "What was our revenue last month?",
            "user_id":      "user_abc123",
            "callback_url": "https://your-app.com/webhooks/analytics-result"
        }

    Response (happy path — metric found):
        HTTP 200
        {
            "status": "completed",
            "thread_id": "<uuid>",
            "result": { "summary": "...", "data": [...], "metric_name": "..." }
        }

    Response (fallback — metric not found, graph suspended):
        HTTP 202
        {
            "status": "pending",
            "thread_id": "<uuid>",
            "message": "Thank you for your question. This metric isn't available yet…"
        }
    """
    import json as _json
    try:
        body = _json.loads(request.body)
    except _json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    print(body)

    message = (body.get("messages") or [{}])[0]

    question = (
        body.get("question")
        or message.get("id", "")
    ).strip()

    user_id = (
        body.get("user_id")
        or message.get("text", {}).get("body", "")
    ).strip()

    callback_url = body.get("callback_url", "").strip()

    user_phone = (
        body.get("user_phone")
        or message.get("from", "")
    ).strip() or None
    print(user_phone)
    if not question:
        return JsonResponse({"error": "'question' is required."}, status=200)
    if not user_id:
        return JsonResponse({"error": "'user_id' is required."}, status=200)
    if not callback_url:
        logging.warning("'callback_url' is required.")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "user_id": user_id,
        "user_phone": user_phone,
        "callback_url": callback_url,
        "thread_id": thread_id,
        "matched_metric": None,
        "classification_confidence": 0.0,
        "cube_query": None,
        "raw_result": None,
        "formatted_result": None,
        "is_resumed": False,
        "fallback_reason": None,
        "resume_data": None,
    }

    try:
        output = graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("graph.invoke failed for thread %s", thread_id)
        return JsonResponse({"error": str(exc)}, status=500)

    # Check whether the graph suspended (interrupt raised)
    # LangGraph surfaces interrupts in the __interrupt__ key of the output
    # when using .invoke(). We detect this and return 202.
    interrupts = output.get("__interrupt__")
    if interrupts:
        interrupt_value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        return JsonResponse(
            {
                "status": "pending",
                "thread_id": thread_id,
                "message": interrupt_value.get(
                    "user_message",
                    "Your request is being processed. You'll be notified when it's ready.",
                ),
            },
            status=202,
        )

    # Happy path — graph ran to completion
    return JsonResponse(
        {
            "status": "completed",
            "thread_id": thread_id,
            "result": output.get("formatted_result"),
        },
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

    Request body (JSON):
        {
            "thread_id": "<uuid from original query response>",
            "metric_id": "new_metric_id",
            "analyst":   "jane@afya.ai"   (optional, for audit logging)
        }

    Response:
        HTTP 200  { "status": "resumed", "thread_id": "…" }
        HTTP 404  { "error": "Thread not found or already completed." }
        HTTP 400  { "error": "…" }
    """
    import json as _json
    try:
        body = _json.loads(request.body)
    except _json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    thread_id = (body.get("thread_id") or "").strip()
    metric_id = (body.get("metric_id") or "").strip()
    analyst = body.get("analyst", "unknown")

    if not thread_id:
        return JsonResponse({"error": "'thread_id' is required."}, status=400)
    if not metric_id:
        return JsonResponse({"error": "'metric_id' is required."}, status=400)

    config = {"configurable": {"thread_id": thread_id}}

    # Verify the thread exists and is suspended
    state_snapshot = graph.get_state(config)
    if state_snapshot is None:
        return JsonResponse({"error": "Thread not found or already completed."}, status=404)

    resume_payload = {
        "metric_id": metric_id,
        "analyst": analyst,
    }

    logger.info(
        "resume: thread=%s metric_id=%s analyst=%s",
        thread_id, metric_id, analyst,
    )

    try:
        graph.invoke(Command(resume=resume_payload), config=config)
    except Exception as exc:
        logger.exception("graph.invoke(resume) failed for thread %s", thread_id)
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({"status": "resumed", "thread_id": thread_id}, status=200)