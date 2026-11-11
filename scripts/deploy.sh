#!/usr/bin/env bash
# scripts/deploy.sh — deploy this checkout's branch. Runs ON the server.
#   production (datahub.afyaanalytics.com):      branch main,    .github/workflows/deploy.yml
#   staging    (datapipelines.afyaanalytics.ai): branch staging, .github/workflows/deploy-staging.yml
# Started by those workflows after a push/merge, or by hand:
#   ~/dashboard/scripts/deploy.sh                  deploy whatever BRANCH (default main) has that we don't
#   BRANCH=staging ~/dashboard/scripts/deploy.sh   on the staging server
#   FULL=1 ~/dashboard/scripts/deploy.sh           also rebuild web even if nothing changed
# The health check uses DOMAIN from the environment, else DOMAIN= in this
# server's .env, else datahub.afyaanalytics.com.
#
# It only touches what the new commits affect:
#   web image code / Dockerfile / requirements   → rebuild + recreate `web`
#   nginx/                                        → rebuild + recreate `nginx`
#   docker-compose.airflow.yaml                   → recreate Airflow (existing image)
#   Dockerfile.airflow, pipelines/requirements.txt → rebuild Airflow image + recreate
#   docker-compose.redash.yaml, redash/           → recreate redash-server/worker/scheduler only
# dags/ and pipelines/ are bind-mounted into Airflow, so a pull is enough for them.
# Other docker-compose.*.yaml files (cube, streamlit, n8n, wandb…) are NOT redeployed
# automatically — the log says so when one of them changed.
#
# The server is a small, CPU-throttled t2.large: builds are slow, so this is meant
# to be run detached (nohup) — the workflow does that and polls the log.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dashboard}"
BRANCH="${BRANCH:-main}"
cd "$REPO_DIR"
if [ -z "${DOMAIN:-}" ] && [ -f .env ]; then
  DOMAIN=$(sed -n 's/^DOMAIN=//p' .env | tail -n 1 | tr -d "\"' \r")
fi
DOMAIN="${DOMAIN:-datahub.afyaanalytics.com}"

log() { echo "== $(date -u +%H:%M:%S) $*"; }

# one deploy at a time
exec 9>"$HOME/.deploy.lock"
flock -n 9 || { log "another deploy is already running — exiting"; exit 75; }

# ── pull ────────────────────────────────────────────────────────────────────
old=$(git rev-parse HEAD)
git fetch -q origin "$BRANCH"
new=$(git rev-parse "origin/$BRANCH")
log "deploying $BRANCH: ${old:0:7} → ${new:0:7}"
if [ "$old" != "$new" ]; then
  # --autostash: logs/dag_processor/latest is tracked and rewritten by Airflow
  git pull --ff-only --autostash origin "$BRANCH"
  [ "$(git rev-parse HEAD)" = "$new" ] || { log "PULL FAILED (server checkout has diverged?)"; exit 1; }
fi
changed=$(git diff --name-only "$old" "$new")
if [ -n "$changed" ]; then
  log "changed files:"; echo "$changed" | sed 's/^/   /'
fi

has() { echo "$changed" | grep -qE "$1"; }

# ── web ─────────────────────────────────────────────────────────────────────
# Everything goes into the web image (COPY . .) except what .dockerignore drops
# and files that only matter to other stacks.
NOT_WEB='^(dags/|pipelines/|plugins/|logs/|redash/|nginx/|\.github/|docs/|scripts/deploy\.sh$|Dockerfile\.airflow|docker-compose\.[^/]+\.ya?ml$|.*\.md$)'
web_changed=$(echo "$changed" | grep -vE "$NOT_WEB" | grep -v '^$' || true)
if [ -n "$web_changed" ] || [ "${FULL:-0}" = 1 ]; then
  log "web: build"
  DOCKER_BUILDKIT=1 docker compose build --progress=plain web
  log "web: recreate"
  docker compose up -d --no-deps web
  restarted_web=1
else
  log "web: no changes"
fi

if has '^nginx/'; then
  log "nginx: build + recreate"
  DOCKER_BUILDKIT=1 docker compose build nginx
  docker compose up -d --no-deps nginx
fi

# ── airflow ─────────────────────────────────────────────────────────────────
if has '^(Dockerfile\.airflow|pipelines/requirements\.txt)'; then
  log "airflow: rebuild image + recreate"
  DOCKER_BUILDKIT=1 docker compose -f docker-compose.airflow.yaml up -d --build
elif has '^docker-compose\.airflow\.yaml$'; then
  log "airflow: recreate (existing image)"
  docker compose -f docker-compose.airflow.yaml up -d --no-build
fi

# ── redash ──────────────────────────────────────────────────────────────────
if has '^(docker-compose\.redash\.yaml$|redash/)'; then
  # Use the project the running containers belong to; --no-deps so Compose
  # leaves redash-postgres / redash-redis alone; --force-recreate so a changed
  # bind-mounted file (redash/query_runner/…) is actually reloaded.
  project=$(docker inspect redash-server --format '{{index .Config.Labels "com.docker.compose.project"}}' 2>/dev/null || echo dashboard-redash)
  log "redash: recreate app containers (project $project)"
  docker compose -p "$project" -f docker-compose.redash.yaml --env-file .env \
    up -d --no-deps --force-recreate redash-server redash-worker redash-scheduler
fi

others=$(echo "$changed" | grep -E '^docker-compose\.[^/]+\.ya?ml$' \
         | grep -vE '^docker-compose\.(airflow|redash)\.ya?ml$' || true)
if [ -n "$others" ]; then
  log "NOTE: changed but not auto-deployed (apply by hand if needed):"; echo "$others" | sed 's/^/   /'
fi

# ── health ──────────────────────────────────────────────────────────────────
# Django's DB container has restart: no — a Docker restart leaves it down and
# the site 502s. Make sure it's up.
if [ "$(docker inspect -f '{{.State.Running}}' postgres-container 2>/dev/null)" = false ]; then
  log "postgres-container was stopped — starting it"
  docker start postgres-container
  [ -n "${restarted_web:-}" ] || docker restart dashboard-web-1
fi

log "health: waiting for https://$DOMAIN (via this server) to answer"
code=000
for i in $(seq 1 60); do   # Django starts slowly on this box (~4 min)
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 \
           --resolve "$DOMAIN:443:127.0.0.1" "https://$DOMAIN/" || true)
  case "$code" in 2??|3??) break ;; esac
  sleep 10
done
if ! [[ "$code" =~ ^[23] ]]; then
  log "HEALTH CHECK FAILED: HTTP $code"
  docker logs --tail 40 dashboard-web-1 2>&1 | tail -40
  exit 1
fi
log "health: HTTP $code"

docker ps --format '{{.Names}}\t{{.Status}}' | sort | grep -iE 'unhealthy|restarting|exited' \
  && log "WARNING: containers above are not healthy" || true
log "DEPLOY OK ${new:0:7}"
