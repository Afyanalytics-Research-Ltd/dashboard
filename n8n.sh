#!/usr/bin/env bash
# Create an n8n API key without the UI. Sets up the owner first if n8n has none (e.g. after user-management:reset).
set -euo pipefail
BASE="${N8N:-http://localhost:5679}"; LABEL="${LABEL:-datahub-django}"
JAR=$(mktemp); trap 'rm -f "$JAR"' EXIT
H=(-H "Content-Type: application/json" -H "browser-id: cli-$(date +%s)")
REST=""; for p in "" /n8n; do curl -s "$BASE$p/rest/settings" | grep -q '"data"' && { REST="$BASE$p/rest"; break; }; done
[ -n "$REST" ] || { echo "n8n REST API not found at $BASE"; exit 1; }; echo "n8n REST API: $REST"
post() { curl -s -o /tmp/n8n_resp -w '%{http_code}' -b "$JAR" -c "$JAR" "${H[@]}" -d "$2" "$REST$1"; }
msg()  { python3 -c 'import json;d=json.load(open("/tmp/n8n_resp"));print(d.get("message") or d)' 2>/dev/null || head -c 200 /tmp/n8n_resp; }

read -rp "n8n email: " EMAIL; read -rsp "n8n password (8+ chars, 1 number, 1 uppercase): " PW; echo
login=$(python3 -c 'import json,sys;print(json.dumps({"emailOrLdapLoginId":sys.argv[1],"email":sys.argv[1],"password":sys.argv[2]}))' "$EMAIL" "$PW")
code=$(post /login "$login")
if [ "$code" != 200 ]; then
  echo "login: HTTP $code - $(msg)"
  echo "Trying owner setup (works only when n8n has no owner, e.g. after user-management:reset)..."
  read -rp "first name: " FN; read -rp "last name: " LN
  setup=$(python3 -c 'import json,sys;print(json.dumps({"email":sys.argv[1],"password":sys.argv[2],"firstName":sys.argv[3],"lastName":sys.argv[4]}))' "$EMAIL" "$PW" "$FN" "$LN")
  code=$(post /owner/setup "$setup")
  [ "$code" = 200 ] || { echo "owner setup: HTTP $code - $(msg)"; exit 1; }
  echo "owner created and logged in as $EMAIL"
else
  echo "logged in"
fi

scopes=$(curl -s -b "$JAR" "${H[@]}" "$REST/api-keys/scopes" | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin)["data"]))' 2>/dev/null || echo '[]')
req=$(python3 -c 'import json,sys;print(json.dumps({"label":sys.argv[1],"scopes":json.loads(sys.argv[2]),"expiresAt":None}))' "$LABEL" "$scopes")
code=$(post /api-keys "$req")
[ "$code" = 200 ] || { echo "create key: HTTP $code - $(msg)"; exit 1; }
KEY=$(python3 -c 'import json;d=json.load(open("/tmp/n8n_resp"))["data"];print(d.get("rawApiKey") or d.get("apiKey"))'); rm -f /tmp/n8n_resp
echo "check: HTTP $(curl -s -o /dev/null -w '%{http_code}' -H "X-N8N-API-KEY: $KEY" "${REST%/rest}/api/v1/workflows?limit=1") (200 = key works)"
echo "N8N_API_KEY=$KEY"
