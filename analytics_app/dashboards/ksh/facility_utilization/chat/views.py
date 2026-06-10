import json
import os
import re
import threading
from pathlib import Path

import httpx
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# facility_utilization/ — one level up from chat/views.py
_NOTICES_DIR = Path(__file__).resolve().parent.parent
_CONTEXT_DIR  = Path(__file__).resolve().parent.parent / "context"

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
    for name in ["current_notices_KSH.json", "current_notices.json"]:
        path = _NOTICES_DIR / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"facility": "KSH", "date": "", "count": 0, "notices": []}


_ctx_lock = threading.Lock()
_ctx_mtime: float = 0.0
_CONTEXT: dict = {}
_DOMAIN_KEYWORDS: list = []


def _load_context() -> tuple:
    context: dict = {}
    keywords: list = []
    for f in sorted(_CONTEXT_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        context[f.stem] = text
        for line in text.splitlines():
            if line.startswith("**keywords:**"):
                kwds = [k.strip() for k in line.replace("**keywords:**", "").split(",")]
                keywords.append((kwds, f.stem))
                break
    return context, keywords


def _get_context() -> tuple:
    """Return (CONTEXT, DOMAIN_KEYWORDS), reloading from disk if any file has changed."""
    global _CONTEXT, _DOMAIN_KEYWORDS, _ctx_mtime
    try:
        latest = max((_CONTEXT_DIR.glob("*.md")), default=None)
        latest = latest.stat().st_mtime if latest else 0.0
    except Exception:
        latest = 0.0
    if latest <= _ctx_mtime:
        return _CONTEXT, _DOMAIN_KEYWORDS
    with _ctx_lock:
        try:
            latest = max((f.stat().st_mtime for f in _CONTEXT_DIR.glob("*.md")), default=0.0)
        except Exception:
            latest = 0.0
        if latest <= _ctx_mtime:
            return _CONTEXT, _DOMAIN_KEYWORDS
        _CONTEXT, _DOMAIN_KEYWORDS = _load_context()
        _ctx_mtime = latest
    return _CONTEXT, _DOMAIN_KEYWORDS


# Warm the cache at startup
_get_context()


def _kw_match(text: str, keywords: list) -> int:
    return sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text))


def _relevant_context(question: str, notices: list) -> str:
    context, domain_keywords = _get_context()
    q = question.lower()
    scores: dict = {}
    for kwds, key in domain_keywords:
        n = _kw_match(q, kwds)
        if n > 0:
            scores[key] = n
    keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    if not keys:
        seen: set = set()
        for notice in notices:
            t = notice.get("title", "").lower()
            for kwds, key in domain_keywords:
                if key not in seen and _kw_match(t, kwds) > 0:
                    keys.append(key)
                    seen.add(key)

    if not keys:
        keys = [key for _, key in domain_keywords]

    parts: list = []
    char_count = 0
    for k in keys:
        if k not in context:
            continue
        block = context[k]
        # Always include the first (highest-scoring) match even if large.
        # Subsequent blocks are added only while under 16,000 chars total.
        if char_count > 0 and char_count + len(block) > 16000:
            break
        parts.append(block)
        char_count += len(block)
    return "\n\n---\n\n".join(parts)


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
        f"1. Only cite metric values (percentages, visit counts, thresholds) that appear VERBATIM in the context. Never calculate, extrapolate, or invent figures. When a question asks about something the investigation has not confirmed, use Status: No Evidence Found and state only what the investigation DID establish on the nearest related topic — do not fill gaps with plausible reasoning or general knowledge.\n"
        f"2. Present findings as analytical outputs, not operational prescriptions. Do not recommend hiring, redistribution, or specific management actions. State what the data shows and who should review it.\n"
        f"3. Use doctor usernames exactly as written in context (e.g. eawando, jogutu). Never swap roles.\n"
        f"4. NEVER tell the user to 'check the dashboard' — you ARE the analytics layer. Answer directly from context.\n"
        f"5. When answering 'why' questions, lead with the strongest investigation finding, not the data limitation.\n"
        f"6. Present confirmed investigation findings as facts: 'the data shows', 'the investigation confirmed'.\n"
        f"7. Distinguish between investigation findings and unresolved details. When the analysis has identified the mechanism driving a metric, present that mechanism as a completed finding. Frame remaining unknowns as narrower operational questions, not evidence that the investigation is incomplete.\n"
        f"8. Format EVERY response — first and all follow-up — using this exact structure. Never deviate.\n\n"
        f"Status: [WATCH / CRITICAL / OK / Attention Required / No Evidence Found]\n\n"
        f"[Section label — choose: Key finding / Answer / Likely impact / Known limitation]\n"
        f"[One sentence only. Plain text, no bullet. State the most important insight.]\n\n"
        f"[Section label — choose: Evidence / Known facts / Related observations / Impact]\n"
        f"• [Metric or fact — one line]\n"
        f"• [Second — one line, only if genuinely distinct; omit if not needed]\n"
        f"• [Third — only if essential]\n\n"
        f"Strict rules for this format:\n"
        f"- Status is always first. It reflects what the data says about the issue, not whether the response is complete.\n"
        f"- The first section (Key finding / Answer) is always a single plain-text sentence — no bullet.\n"
        f"- The second section (Evidence / Known facts / Related observations) is always bullet points — no prose.\n"
        f"- Never repeat a fact that appeared in the first section inside the second section.\n"
        f"- Never mix evidence and conclusions in the same section.\n"
        f"- No paragraphs, no narrative, no reasoning shown, no root-cause analysis, no trends, no recommendations.\n\n"
        f"End every response with:\nFOLLOW-UPS:\n• [most valuable next question]\n• [second most valuable next question]\n"
        f"9. For broad questions: summarize ONLY what the investigation context explicitly covers.\n"
        f"10. For follow-up questions: apply the same format as Rule 8. Do not repeat any metric or finding already stated in a prior response in this conversation — each follow-up adds only new information. Answer only what was asked.\n"
        f"11. Findings that rule out explanations are completed analytical outcomes — state them explicitly.\n"
        f"12. If the question cannot be answered from context or active alerts, use this structure:\n"
        f"Status: No Evidence Found\n\n"
        f"What's available\n"
        f"• [Closest confirmed finding from the investigation on this topic]\n"
        f"• [Second related finding if available]\n"
        f"Do not say 'the investigation did not test', 'the data does not show', or 'we never investigated'. Do not explain what is missing. State only what IS available. Do not attempt to answer from general medical knowledge. No Evidence Found applies only when investigation data is genuinely absent — do not use it when a question contains action words like 'redistribution', 'options', or 'workload'; if the data exists in context, report it directly.\n"
        f"13. Do NOT mention specific doctor names, departure events, or historical personnel changes unless the question explicitly asks about a named person or a departure. State structural and pattern findings without attributing them to individuals. This applies even when an active alert contains a doctor's name — describe the concentration pattern without naming the individual. Only state a name if the user explicitly asks 'which doctor' or names someone themselves."
    )


def chat_page(request):
    request.session.pop("chat_history", None)  # always start fresh on page load
    notices_data = _load_notices()
    return render(request, "facility_utilization/chat/chat.html", {
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

        ctx          = _relevant_context(message, notices)
        system_msg   = _build_system_prompt(facility, date, notices)
        history      = request.session.get("chat_history", [])

        msgs = [{"role": "system", "content": system_msg}]
        if ctx:
            msgs.append({"role": "system", "content": f"INVESTIGATION CONTEXT:\n{ctx}"})
        msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": message})

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
