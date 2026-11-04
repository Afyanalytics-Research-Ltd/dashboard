#!/usr/bin/env bash
# Copy the Airflow DAGs and pipeline scripts from the datapipelines repo into
# this repo, so the dashboard's Airflow (docker-compose.airflow.yaml) runs them.
#
#   scripts/sync_datapipelines.sh [--with-state] [SRC]
#
#   SRC           datapipelines checkout (default: ~/datapipelines)
#   --with-state  also copy runtime files that are NOT in git: pipelines .env,
#                 Snowflake/Google keys, migration state (.migration_state/,
#                 .migration_*.json, watermarks, progress files) and inputs.
#
# Layout produced (paths inside the scripts stay exactly as in datapipelines):
#   dags/                         <- SRC/dags            (DAG files, pipelines_common.py, jsons/)
#   config/airflow_local_settings.py <- SRC/config/...   (Sentry + cluster policies)
#   plugins/                      <- SRC/plugins
#   pipelines/                    <- every other git-tracked file of SRC
#   pipelines/dags                -> ../dags             (symlink; scripts import ROOT/dags)
#   pipelines/config/airflow_local_settings.py -> ../../config/airflow_local_settings.py
#
# Code is copied from SRC's working tree (so uncommitted edits come along),
# limited to files git tracks there plus EXTRA_UNTRACKED below. Re-running is
# safe; it never deletes anything here.
#
# STATE: the migration scripts guard concurrent runs with flock on files in
# the state dir. Two Airflow stacks with separate copies of the state do NOT
# see each other's locks, so only one stack may run these DAGs at a time. At
# cutover: stop the datapipelines stack, then run this with --with-state.
set -euo pipefail

WITH_STATE=0
SRC=""
for arg in "$@"; do
  case "$arg" in
    --with-state) WITH_STATE=1 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) SRC="$arg" ;;
  esac
done
SRC="${SRC:-$HOME/datapipelines}"
DST="$(cd "$(dirname "$0")/.." && pwd)"

[ -d "$SRC/.git" ] || { echo "Not a git checkout: $SRC" >&2; exit 1; }
[ -f "$SRC/dags/pipelines_common.py" ] || { echo "No dags/pipelines_common.py in $SRC" >&2; exit 1; }

# Not (yet) committed in datapipelines but needed: a module a DAG imports, its
# tests (tests/ is added whole below) and the Airflow API notes. Untracked
# one-off scripts there (fix_*.py, ...) are deliberately not copied.
EXTRA_UNTRACKED=(v3_to_snowflake_sync.py AIRFLOW_API.md)

# Tracked in datapipelines but not code: Docker/infra for its own stack, and
# migration state (copied with --with-state instead, never committed here).
is_skipped() {
  case "$1" in
    dags/*|plugins/*|config/*) return 0 ;;                 # handled separately
    Dockerfile|docker-compose*.yaml|nginx/*|nginx_templates/*|.gitignore) return 0 ;;
    .migration_state/*|.migration_*|.progress.json*) return 0 ;;
    *.bak|*.bak_*|*.bak.*|*.tmp.*) return 0 ;;
  esac
  return 1
}

echo "Source: $SRC ($(git -C "$SRC" rev-parse --abbrev-ref HEAD) @ $(git -C "$SRC" rev-parse --short HEAD))"
dirty=$(git -C "$SRC" status --porcelain --untracked-files=no | wc -l)
[ "$dirty" -eq 0 ] || echo "  note: $dirty tracked file(s) have uncommitted changes in $SRC; the working-tree versions are copied"

RSYNC_COMMON=(-rlt --exclude=__pycache__/ --exclude='*.pyc' --exclude='*.tmp.*' --exclude='*.bak_*' --exclude='*.bak')

# 1. DAGs (+ jsons/), Airflow local settings, plugins
mkdir -p "$DST/dags" "$DST/config" "$DST/plugins"
rsync "${RSYNC_COMMON[@]}" "$SRC/dags/" "$DST/dags/"
cp "$SRC/config/airflow_local_settings.py" "$DST/config/airflow_local_settings.py"
rsync "${RSYNC_COMMON[@]}" "$SRC/plugins/" "$DST/plugins/"

# 2. Pipeline scripts: git-tracked files of SRC (working-tree copy)
mkdir -p "$DST/pipelines"
list=$(mktemp); trap 'rm -f "$list"' EXIT
git -C "$SRC" ls-files -z | while IFS= read -r -d '' f; do
  is_skipped "$f" && continue
  [ -e "$SRC/$f" ] || continue                       # deleted in the working tree
  printf '%s\n' "$f"
done > "$list"
for f in "${EXTRA_UNTRACKED[@]}"; do
  [ -e "$SRC/$f" ] && printf '%s\n' "$f" >> "$list"
done
git -C "$SRC" ls-files --others --exclude-standard -- 'tests/*.py' >> "$list"
rsync -lt --files-from="$list" "$SRC/" "$DST/pipelines/"
echo "  copied $(wc -l < "$list") pipeline files, $(find "$DST/dags" -maxdepth 1 -name '*.py' | wc -l) DAG .py files"

# 3. Symlinks so the scripts see the same tree they do in datapipelines
ln -sfn ../dags "$DST/pipelines/dags"
mkdir -p "$DST/pipelines/config"
ln -sfn ../../config/airflow_local_settings.py "$DST/pipelines/config/airflow_local_settings.py"

# 4. Runtime state + secrets (gitignored here)
if [ "$WITH_STATE" -eq 1 ]; then
  echo "Copying runtime state + secrets ..."
  rsync -rlt --prune-empty-dirs \
    --exclude='*.bak*' --exclude='*.tmp' --exclude='*.tmp.*' --exclude='*.lock' --exclude=__pycache__/ \
    --include=/.env \
    --include=/service_account.json --include=/gdrive-sa.json \
    --include=/config/ --include=/config/rsa_key.p8 --include=/config/rsa_key.pub --include=/config/gdrive-sa.json \
    --include='/.migration_state/***' --include='/.page_progress/***' --include='/state/***' \
    --include='/input/***' --include='/schema_cache/***' \
    --include=/mapping/ --include='/mapping/softdelete/***' \
    --include='/*.json' \
    --exclude='*' \
    "$SRC/" "$DST/pipelines/"
  echo "  state copied: $(du -sh "$DST/pipelines/.migration_state" 2>/dev/null | cut -f1) in .migration_state/"
fi
echo "Done."
