#!/usr/bin/env bash
# wandb.sh - create a W&B Server API key without the UI. Run on the server.
set -euo pipefail
B="${WANDB:-http://127.0.0.1:8090}"
JAR=$(mktemp); trap 'rm -f "$JAR" /tmp/wb_r' EXIT

# W&B rejects account changes unless Origin matches its configured HOST (WANDB_HOST_URL)
ORIGIN="${WANDB_ORIGIN:-$(docker exec wandb-local printenv HOST)}"; ORIGIN="${ORIGIN%/}"
echo "using Origin: $ORIGIN"
J=(-H 'Content-Type: application/json' -H "Origin: $ORIGIN" -H "Referer: $ORIGIN/")

login() { curl -s -o /tmp/wb_r -w '%{http_code}' -c "$JAR" -b "$JAR" "${J[@]}" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"email":sys.argv[1],"password":sys.argv[2]}))' "$1" "$2")" "$B/oidc/login"; }

read -rp "W&B admin email [local@wandb.com]: " EMAIL; EMAIL=${EMAIL:-local@wandb.com}
read -rsp "password [Enter = 'perceptron', the fresh-server default]: " PW; echo; PW=${PW:-perceptron}
code=$(login "$EMAIL" "$PW"); [ "$code" = 200 ] || { echo "login failed: HTTP $code $(head -c 200 /tmp/wb_r)"; exit 1; }
echo "logged in as $EMAIL"

if [ "$EMAIL" = local@wandb.com ]; then   # replace the default admin: its password is public
  read -rp "your admin email: " NE; read -rsp "your admin password: " NP; echo
  body=$(python3 -c 'import json,sys;print(json.dumps([{"email":sys.argv[1],"password":sys.argv[2]},{"email":"local@wandb.com","delete":True}]))' "$NE" "$NP")
  code=$(curl -s -o /tmp/wb_r -w '%{http_code}' -b "$JAR" -c "$JAR" -X PUT "${J[@]}" -d "$body" "$B/oidc/users")
  [ "$code" = 200 ] || { echo "changing admin failed: HTTP $code $(head -c 200 /tmp/wb_r)"; exit 1; }
  code=$(login "$NE" "$NP"); [ "$code" = 200 ] || { echo "login as $NE failed: HTTP $code $(head -c 200 /tmp/wb_r)"; exit 1; }
  echo "admin is now $NE"
fi

resp=$(curl -s -b "$JAR" -X POST "${J[@]}" \
  -d '{"query":"mutation { generateApiKey(input:{description:\"datahub-django\"}) { secretApiKey } }"}' "$B/graphql")
KEY=$(echo "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"]["generateApiKey"]["secretApiKey"])') \
  || { echo "key creation failed: $(echo "$resp" | head -c 300)"; exit 1; }
echo "check: $(curl -s -u "api:$KEY" -X POST -H 'Content-Type: application/json' -d '{"query":"{ viewer { username admin } }"}' "$B/graphql")"
echo
echo "WANDB_API_KEY=$KEY"
