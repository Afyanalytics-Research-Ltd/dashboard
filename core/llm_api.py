"""Internal model endpoints for other containers (n8n, Airflow) on dashboard-net.

All free and local (fastembed on CPU):
  POST /api/llm/v1/embeddings  OpenAI-compatible {model?, input, input_type?: "query"|"passage"}
  POST /api/llm/v1/rerank      {query, documents[]} -> {results: [{index, score}]} sorted
  POST /api/llm/v1/kb/search   {query, top_k?, sources?} -> support KB hybrid search + rerank

Auth: `Authorization: Bearer $LLM_INTERNAL_TOKEN`. With the variable unset the endpoints
answer 503, so nothing is exposed by accident.
"""
from __future__ import annotations

import hmac
import json
import os
import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


def _auth(request):
    token = os.getenv("LLM_INTERNAL_TOKEN", "").strip()
    if not token:
        return JsonResponse({"error": "LLM_INTERNAL_TOKEN is not configured"}, status=503)
    got = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(got, token):
        return JsonResponse({"error": "unauthorized"}, status=401)
    return None


def _body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except ValueError:
        return {}


@csrf_exempt
@require_POST
def embeddings(request):
    if (err := _auth(request)):
        return err
    from afya_backends import llm
    b = _body(request)
    inp = b.get("input")
    texts = [inp] if isinstance(inp, str) else [str(t) for t in (inp or [])]
    if not texts:
        return JsonResponse({"error": "input is required"}, status=400)
    if len(texts) > 256:
        return JsonResponse({"error": "at most 256 inputs per call"}, status=400)
    model = b.get("model") if "/" in str(b.get("model") or "") else llm.local_embed_model()
    vecs = llm.local_embed(texts, query=b.get("input_type") == "query", model_name=model)
    return JsonResponse({"object": "list", "model": model,
                         "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                         "usage": {"prompt_tokens": 0, "total_tokens": 0}})


@csrf_exempt
@require_POST
def rerank(request):
    if (err := _auth(request)):
        return err
    from afya_backends import llm
    b = _body(request)
    q, docs = str(b.get("query") or ""), [str(d) for d in (b.get("documents") or [])]
    if not q or not docs:
        return JsonResponse({"error": "query and documents are required"}, status=400)
    scores = llm.local_rerank(q, docs[:100])
    res = sorted(({"index": i, "score": s} for i, s in enumerate(scores)), key=lambda r: -r["score"])
    top_n = int(b.get("top_n") or len(res))
    return JsonResponse({"model": llm.local_rerank_model(), "results": res[:top_n]})


@csrf_exempt
@require_POST
def kb_search(request):
    if (err := _auth(request)):
        return err
    from support_kb import store
    b = _body(request)
    q = str(b.get("query") or "").strip()[:1000]
    if not q:
        return JsonResponse({"error": "query is required"}, status=400)
    t = time.monotonic()
    hits = store.search(q, top_k=max(1, min(int(b.get("top_k") or 6), 20)),
                        sources=b.get("sources") or None)
    return JsonResponse({"query": q, "ms": int((time.monotonic() - t) * 1000), "results": hits})
