"""Build the "HMIS Support Chat (RAG)" n8n workflow + its credentials as import files.

On a server (secrets from ~/dashboard/.env), see support_kb/install_n8n_support_chat.sh:
    set -a; . ~/dashboard/.env; set +a
    python3 support_kb/n8n_support_chat.py /tmp/n8n-support
    docker cp /tmp/n8n-support n8n-container:/tmp/ && docker exec n8n-container sh -c \
      'n8n import:credentials --input=/tmp/n8n-support/credentials.json &&
       n8n import:workflow --input=/tmp/n8n-support/workflow.json &&
       n8n publish:workflow --id=afyaSupportChat1'
    docker restart n8n-container

Pipeline (all free: Groq free tier + local fastembed in the Django web container):
  hosted chat (basic auth) / POST webhook (X-Support-Token)
  -> normalise -> load last turns (support-kb-db) -> if follow-up: Groq rewrites it into a
     standalone question -> web /api/llm/v1/kb/search (vector + keyword, RRF, cross-encoder
     rerank) -> Groq answers from the numbered sources only, citing [n] -> save both turns
  -> reply {output, sources}

Ids are fixed, so re-importing updates the same credentials and workflow in place.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

NAME = "HMIS Support Chat (RAG)"
WEB = os.getenv("SUPPORT_WEB_URL", "http://web:8000")
ANSWER_MODEL = os.getenv("SUPPORT_ANSWER_MODEL", "openai/gpt-oss-120b")
REWRITE_MODEL = os.getenv("SUPPORT_REWRITE_MODEL", "openai/gpt-oss-20b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are the support assistant for Afya Analytics HMIS 3.0, a hospital management system used by hospital staff (receptionists, nurses, doctors, lab, pharmacy, cashiers, store keepers, administrators) and by implementation/support staff.

Answer ONLY from the numbered SOURCES provided with the question. Rules:
- Give the practical answer first: the menu path (e.g. `Setup > HIS Client Settings > Facilities`), the exact button and field names in **bold**, and numbered steps when there are several.
- Cite every fact with its source number in square brackets, like [1] or [2][4]. Never cite a number that is not in SOURCES.
- If the answer is about a missing menu, button or "403/not authorised", name the permission code and the setting from the sources, and say an administrator must grant it.
- If the SOURCES do not contain the answer, say plainly that you could not find it in the HMIS documentation, then suggest who can help (their system administrator, or Afya Analytics support with the page address, record number, time and a screenshot). Do not guess, and never invent menu paths, permission codes, settings or endpoints.
- Backend API endpoints are for implementation/support staff; mention them only when the user asks about the API, integrations or technical errors.
- Never ask for or repeat patient data, passwords, tokens or keys.
- Be concise: at most ~200 words unless the user asks for detail. Use Markdown. Reply in the user's language."""

REWRITE_PROMPT = """Rewrite the user's LAST message as one standalone search query for the HMIS documentation, resolving pronouns and references from the conversation. Keep HMIS terms, screen names, menu names and error text exactly. Output only the query, nothing else."""


# ── node builders ────────────────────────────────────────────────────────────

def node(name, typ, version, pos, params, **extra):
    n = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, NAME + name)), "name": name, "type": typ,
         "typeVersion": version, "position": pos, "parameters": params}
    n.update(extra)
    return n


def http_json(name, pos, url, body_expr, cred_id, cred_name, *, timeout=60000, **extra):
    return node(name, "n8n-nodes-base.httpRequest", 4.2, pos, {
        "method": "POST", "url": url,
        "authentication": "genericCredentialType", "genericAuthType": "httpHeaderAuth",
        "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
        "options": {"timeout": timeout},
    }, credentials={"httpHeaderAuth": {"id": cred_id, "name": cred_name}}, **extra)


def build(creds: dict, chat_webhook_id: str) -> dict:
    pg = {"postgres": {"id": creds["pg"], "name": "Support KB DB"}}
    nodes = [
        node("Support chat", "@n8n/n8n-nodes-langchain.chatTrigger", 1.1, [0, 0], {
            "public": True, "mode": "hostedChat", "authentication": "basicAuth",
            "initialMessages": "Hi! I'm the Afya HMIS support assistant. Ask me how to set something up, "
                               "where to find a screen, or what an error means.",
            "options": {"title": "Afya HMIS Support", "subtitle": "Answers from the HMIS documentation",
                        "responseMode": "lastNode", "inputPlaceholder": "Type your question…"},
        }, webhookId=chat_webhook_id,
            credentials={"httpBasicAuth": {"id": creds["chat_basic"], "name": "Support chat login"}}),
        node("Support API", "n8n-nodes-base.webhook", 2, [0, 220], {
            "httpMethod": "POST", "path": "support-chat", "authentication": "headerAuth",
            "responseMode": "lastNode", "responseData": "firstEntryJson", "options": {},
        }, webhookId=str(uuid.uuid5(uuid.NAMESPACE_URL, NAME + "webhook")),
            credentials={"httpHeaderAuth": {"id": creds["webhook"], "name": "Support API token"}}),
        node("Normalise", "n8n-nodes-base.code", 2, [240, 100], {"jsCode": r"""
const j = $input.first().json;
const message = String(j.chatInput ?? j.body?.message ?? j.message ?? '').trim().slice(0, 2000);
const sessionId = String(j.sessionId ?? j.body?.sessionId ?? ('api-' + Date.now())).slice(0, 120);
return [{ json: { message, sessionId } }];"""}),
        node("Load history", "n8n-nodes-base.postgres", 2.5, [480, 100], {
            "operation": "executeQuery",
            "query": "SELECT role, content FROM (SELECT id, role, content FROM chat_history "
                     "WHERE session_id = $1 ORDER BY id DESC LIMIT 6) t ORDER BY id",
            "options": {"queryReplacement": "={{ [$json.sessionId] }}"},
        }, credentials=pg, alwaysOutputData=True, onError="continueRegularOutput"),
        node("Prepare", "n8n-nodes-base.code", 2, [720, 100], {"jsCode": r"""
const { message, sessionId } = $('Normalise').first().json;
const history = $input.all().map(i => i.json).filter(r => r.role && r.content)
  .map(r => ({ role: r.role, content: String(r.content).slice(0, 1500) }));
const rewriteBody = {
  model: '""" + REWRITE_MODEL + r"""', temperature: 0, max_tokens: 300, reasoning_effort: 'low',
  messages: [{ role: 'system', content: """ + json.dumps(REWRITE_PROMPT) + r""" },
             ...history, { role: 'user', content: message }],
};
return [{ json: { message, sessionId, history, needsRewrite: history.length > 0, rewriteBody } }];"""}),
        node("Follow-up?", "n8n-nodes-base.if", 2.2, [960, 100], {
            "conditions": {"options": {"caseSensitive": True, "typeValidation": "loose", "version": 2},
                           "conditions": [{"id": "c1", "leftValue": "={{ $json.needsRewrite }}", "rightValue": True,
                                           "operator": {"type": "boolean", "operation": "true", "singleValue": True}}],
                           "combinator": "and"},
            "options": {},
        }),
        http_json("Rewrite question (Groq)", [1200, 0], GROQ_URL, "={{ JSON.stringify($json.rewriteBody) }}",
                  creds["groq"], "Groq API", timeout=30000, retryOnFail=True, maxTries=2, waitBetweenTries=1500,
                  onError="continueRegularOutput"),
        node("Search query", "n8n-nodes-base.code", 2, [1440, 100], {"jsCode": r"""
const p = $('Prepare').first().json;
const rewritten = String($json.choices?.[0]?.message?.content ?? '').trim();
const query = (rewritten && rewritten.length < 500) ? rewritten : p.message;
return [{ json: { ...p, query } }];"""}),
        http_json("Search KB", [1680, 100], WEB + "/api/llm/v1/kb/search",
                  "={{ JSON.stringify({ query: $json.query, top_k: 6 }) }}", creds["internal"], "Afya internal LLM",
                  timeout=60000, retryOnFail=True, maxTries=2, waitBetweenTries=2000, onError="continueRegularOutput"),
        node("Build answer request", "n8n-nodes-base.code", 2, [1920, 100], {"jsCode": r"""
const p = $('Search query').first().json;
const hits = ($json.results ?? []).filter(h => h.content);
const sources = hits.map((h, i) => `[${i + 1}] ${h.breadcrumb}${h.url_path ? ' (page ' + h.url_path + ')' : ''}\n${String(h.content).slice(0, 2600)}`).join('\n\n---\n\n');
const user = `SOURCES:\n${sources || '(no matching documentation found)'}\n\nQUESTION: ${p.message}` +
             (p.query !== p.message ? `\n(Interpreted as: ${p.query})` : '');
const answerBody = {
  model: '""" + ANSWER_MODEL + r"""', temperature: 0.2, max_tokens: 1200, reasoning_effort: 'low',
  messages: [{ role: 'system', content: """ + json.dumps(SYSTEM_PROMPT) + r""" }, ...p.history, { role: 'user', content: user }],
};
const sourceList = hits.map((h, i) => ({ n: i + 1, title: h.breadcrumb, page: h.url_path || null, source: h.source, score: h.score }));
return [{ json: { ...p, answerBody, sourceList } }];"""}),
        http_json("Answer (Groq)", [2160, 100], GROQ_URL, "={{ JSON.stringify($json.answerBody) }}",
                  creds["groq"], "Groq API", timeout=60000, retryOnFail=True, maxTries=3, waitBetweenTries=2000,
                  onError="continueRegularOutput"),
        node("Format reply", "n8n-nodes-base.code", 2, [2400, 100], {"jsCode": r"""
const p = $('Build answer request').first().json;
let text = String($json.choices?.[0]?.message?.content ?? '').trim();
if (!text) text = "Sorry — I couldn't answer right now (the AI service didn't respond). Please try again in a minute, or contact your system administrator.";
const cited = new Set([...text.matchAll(/\[(\d+)\]/g)].map(m => Number(m[1])));
const used = p.sourceList.filter(s => cited.has(s.n));
const refs = used.length ? '\n\n**Sources:** ' + used.map(s => `[${s.n}] ${s.title}${s.page ? ' — `' + s.page + '`' : ''}`).join('; ') : '';
return [{ json: { output: text + refs, answer: text, sources: used, sessionId: p.sessionId, message: p.message } }];"""}),
        node("Save history", "n8n-nodes-base.postgres", 2.5, [2640, 100], {
            "operation": "executeQuery",
            "query": "INSERT INTO chat_history (session_id, role, content) VALUES ($1, 'user', $2), ($1, 'assistant', $3)",
            "options": {"queryReplacement": "={{ [$json.sessionId, $json.message, $json.answer] }}"},
        }, credentials=pg, onError="continueRegularOutput", alwaysOutputData=True),
        node("Reply", "n8n-nodes-base.code", 2, [2880, 100], {"jsCode": r"""
const r = $('Format reply').first().json;
return [{ json: { output: r.output, answer: r.answer, sources: r.sources, sessionId: r.sessionId } }];"""}),
    ]
    c = lambda a, b, idx=0: (a, b, idx)  # noqa: E731
    edges = [c("Support chat", "Normalise"), c("Support API", "Normalise"), c("Normalise", "Load history"),
             c("Load history", "Prepare"), c("Prepare", "Follow-up?"), c("Follow-up?", "Rewrite question (Groq)", 0),
             c("Follow-up?", "Search query", 1), c("Rewrite question (Groq)", "Search query"),
             c("Search query", "Search KB"), c("Search KB", "Build answer request"),
             c("Build answer request", "Answer (Groq)"), c("Answer (Groq)", "Format reply"),
             c("Format reply", "Save history"), c("Save history", "Reply")]
    conns: dict = {}
    for a, b, idx in edges:
        outs = conns.setdefault(a, {"main": []})["main"]
        while len(outs) <= idx:
            outs.append([])
        outs[idx].append({"node": b, "type": "main", "index": 0})
    return {"name": NAME, "nodes": nodes, "connections": conns,
            "settings": {"executionOrder": "v1", "saveDataSuccessExecution": "none",
                         "saveDataErrorExecution": "all", "timezone": "Africa/Nairobi"}}


# ── export for `n8n import:*` (the public API needs a per-instance key) ─────

WORKFLOW_ID = "afyaSupportChat1"
CRED_IDS = {"groq": "afyaSuppGroqApi1", "internal": "afyaSuppInternal1", "webhook": "afyaSuppWebhook1",
            "chat_basic": "afyaSuppChatAuth1", "pg": "afyaSuppKbDb0001"}
CHAT_WEBHOOK_ID = "5b7e3c1a-9f4d-4a8e-b1c2-afya5upp0rt1"


def export(out_dir: str) -> None:
    """Write credentials.json + workflow.json for `n8n import:credentials/import:workflow`.
    Secrets come from the environment (the server's ~/dashboard/.env)."""
    e = os.environ
    creds = [
        {"id": CRED_IDS["groq"], "name": "Groq API", "type": "httpHeaderAuth",
         "data": {"name": "Authorization", "value": "Bearer " + e["GROQ_API_KEY"]}},
        {"id": CRED_IDS["internal"], "name": "Afya internal LLM", "type": "httpHeaderAuth",
         "data": {"name": "Authorization", "value": "Bearer " + e["LLM_INTERNAL_TOKEN"]}},
        {"id": CRED_IDS["webhook"], "name": "Support API token", "type": "httpHeaderAuth",
         "data": {"name": "X-Support-Token", "value": e["SUPPORT_WEBHOOK_TOKEN"]}},
        {"id": CRED_IDS["chat_basic"], "name": "Support chat login", "type": "httpBasicAuth",
         "data": {"user": "support", "password": e["SUPPORT_CHAT_PASSWORD"]}},
        {"id": CRED_IDS["pg"], "name": "Support KB DB", "type": "postgres",
         "data": {"host": "support-kb-db", "port": 5432, "database": "support_kb", "user": "support_kb",
                  "password": e["SUPPORT_KB_DB_PASSWORD"], "ssl": "disable", "allowUnauthorizedCerts": False,
                  "sshTunnel": False, "maxConnections": 5}},
    ]
    wf = build(CRED_IDS, CHAT_WEBHOOK_ID)
    wf.update({"id": WORKFLOW_ID, "active": False, "pinData": {}})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "credentials.json").write_text(json.dumps(creds))
    (out / "workflow.json").write_text(json.dumps(wf, indent=1))
    print(f"wrote {out}/credentials.json ({len(creds)}) and {out}/workflow.json ({len(wf['nodes'])} nodes)")


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "/tmp/n8n-support")
