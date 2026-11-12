#!/bin/bash
# Install / update the HMIS Support Chat (RAG) workflow on this server's n8n.
# Run on the server from ~/dashboard. Needs in .env: GROQ_API_KEY, LLM_INTERNAL_TOKEN,
# SUPPORT_KB_DB_PASSWORD, SUPPORT_CHAT_PASSWORD, SUPPORT_WEBHOOK_TOKEN.
# Restarts n8n-container (a few seconds) so the published workflow's webhooks register.
set -euo pipefail
cd ~/dashboard
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
# Read only the variables we need (the .env has values a shell can't source).
for v in GROQ_API_KEY LLM_INTERNAL_TOKEN SUPPORT_KB_DB_PASSWORD SUPPORT_CHAT_PASSWORD SUPPORT_WEBHOOK_TOKEN; do
  val=$(grep -E "^$v=" .env | tail -1 | cut -d= -f2- | sed -e "s/^['\"]//" -e "s/['\"]$//")
  [ -n "$val" ] || { echo "missing $v in .env" >&2; exit 1; }
  export "$v=$val"
done
python3 support_kb/n8n_support_chat.py "$OUT/n8n-support"
chmod -R a+r "$OUT"
docker exec n8n-container rm -rf /tmp/n8n-support
docker cp "$OUT/n8n-support" n8n-container:/tmp/n8n-support
docker exec n8n-container sh -c '
  n8n import:credentials --input=/tmp/n8n-support/credentials.json &&
  n8n import:workflow --input=/tmp/n8n-support/workflow.json &&
  n8n publish:workflow --id=afyaSupportChat1; rc=$?; rm -rf /tmp/n8n-support; exit $rc'
docker restart n8n-container >/dev/null
echo "Support chat installed. n8n restarting."
