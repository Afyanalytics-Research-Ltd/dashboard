import json
import os
import re
from pathlib import Path

import httpx
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST

from . import prompt_builder, router, sql_executor

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# facility_utilization/ — two levels up from chat/views.py
_PRIVATE = Path(__file__).resolve().parent.parent

# Inline SVG avatars — no static file dependency
_BOT_SVG = mark_safe(
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="7" y="5" width="10" height="8" rx="2" fill="white"/>'
    '<circle cx="9.5" cy="8.5" r="1.5" fill="#003467"/>'
    '<circle cx="14.5" cy="8.5" r="1.5" fill="#003467"/>'
    '<rect x="10" y="11" width="4" height="1.2" rx="0.6" fill="#003467"/>'
    '<rect x="11" y="2" width="2" height="3" rx="1" fill="white"/>'
    '<circle cx="12" cy="2" r="1" fill="#7FB3E0"/>'
    '<rect x="9" y="13" width="6" height="5" rx="1" fill="white"/>'
    '<rect x="5" y="14" width="3" height="3" rx="1" fill="white"/>'
    '<rect x="16" y="14" width="3" height="3" rx="1" fill="white"/>'
    '</svg>'
)

_USER_SVG = mark_safe(
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="12" cy="8" r="4" fill="#6B8CAE"/>'
    '<path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="#6B8CAE" stroke-width="2" stroke-linecap="round"/>'
    '</svg>'
)


def _load_notices() -> dict:
    for name in [f"current_notices_KSH.json", "current_notices.json"]:
        path = _PRIVATE / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"facility": "KSH", "date": "", "count": 0, "notices": []}


def chat_page(request):
    request.session.pop("chat_history", None)  # always start fresh on page load
    notices_data = _load_notices()
    return render(request, "chat/chat.html", {
        "notices":  notices_data.get("notices", []),
        "facility": notices_data.get("facility", "KSH"),
        "date":     notices_data.get("date", ""),
        "history":  [],
        "bot_svg":  _BOT_SVG,
        "user_svg": _USER_SVG,
        "prefill":  request.GET.get("q", ""),
    })


@require_POST
def chat_api(request):
    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()
        if not message:
            return JsonResponse({"error": "Empty message"}, status=400)

        groq_key = os.getenv("GROQ_API", "")
        if not groq_key:
            return JsonResponse({"reply": "GROQ_API key not configured in .env"})

        notices_data = _load_notices()
        notices  = notices_data.get("notices", [])
        facility = notices_data.get("facility", "KSH")
        date     = notices_data.get("date", "")

        history      = request.session.get("chat_history", [])
        route_result = router.route(message)
        sql_result   = sql_executor.execute(message, route_result) if route_result.get("use_sql") else None
        msgs         = prompt_builder.build_messages(message, route_result, notices, facility, date, history, sql_result)

        with httpx.Client() as client:
            resp = client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": msgs,
                      "max_tokens": 512, "temperature": 0.1},
                timeout=30,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]

        # Strip FOLLOW-UPS block before storing — chips are a UI feature, not history
        clean_reply = re.sub(r'\s*FOLLOW-UPS:[\s\S]*$', '', reply, flags=re.IGNORECASE).strip()
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": clean_reply},
        ]
        request.session["chat_history"] = history[-12:]

        return JsonResponse({"reply": reply})

    except httpx.TimeoutException:
        return JsonResponse({"reply": "Request timed out — try again in a moment."})
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("error", {}).get("message", e.response.text[:200])
        except Exception:
            detail = e.response.text[:200]
        return JsonResponse({"reply": f"Groq error {e.response.status_code}: {detail}"})
    except Exception as e:
        return JsonResponse({"reply": f"Something went wrong: {str(e)}"})


@require_POST
def clear_history(request):
    request.session.pop("chat_history", None)
    return JsonResponse({"status": "cleared"})
